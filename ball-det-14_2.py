import cv2
import torch
import argparse
import os
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from enum import Enum

# --- CONFIGURATION BLOCK ---
# --- Paths ---
BALL_MODEL_PATH = "ball.pth"
PLAYER_MODEL_PATH = "player.pth" # Requires the player model
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v14_2.mp4"
PREDICTION_FILE_PATH = "predictions_v14_2.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.7
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] # 1: goalkeeper, 2: player, 3: referee
FPS = 25

# --- V14.1 Core Tracker Parameters ---
OPTICAL_FLOW_GAP = 3
OUTLIER_POSITION_THRESHOLD = 40.0
OUTLIER_VELOCITY_THRESHOLD = 40.0
OUTLIER_HISTORY_FRAMES = 4

# --- V11 Context & State Machine Parameters ---
EXCLUSION_CONFIDENCE_PENALTY = 0.2
ACTION_ZONE_HEIGHT_PERCENT = 0.15
ACTION_ZONE_WIDTH_EXPANSION_PERCENT = 0.1
UPWARD_VELOCITY_THRESHOLD = -25.0
ON_GROUND_CONFIRMATION_THRESHOLD = 2
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 150.0

# --- V12 Physics Model Parameters ---
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.57

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2

# --- Helper Functions from V11 ---
def is_point_in_boxes(point, boxes):
    return np.any((point[0] >= boxes[:, 0]) & (point[0] <= boxes[:, 2]) & (point[1] >= boxes[:, 1]) & (point[1] <= boxes[:, 3]))

def is_point_in_action_zones(point, player_boxes):
    for box in player_boxes:
        x1, y1, x2, y2 = box
        box_height, box_width = y2 - y1, x2 - x1
        action_y_start = y2 - (box_height * ACTION_ZONE_HEIGHT_PERCENT)
        width_expansion = box_width * ACTION_ZONE_WIDTH_EXPANSION_PERCENT
        action_x_start, action_x_end = x1 - width_expansion, x2 + width_expansion
        if (action_x_start <= point[0] <= action_x_end) and (action_y_start <= point[1] <= y2):
            return True
    return False

# --- Physics-Aware Kalman Filter from V12 ---
class PhysicsKalmanFilter:
    def __init__(self, dt=1.0, gravity_y=0.0):
        self.dt = dt
        self.gravity_y = gravity_y
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.5
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((4, 1))
        self.P = np.eye(4) * 100

    def predict(self):
        self.x_hat = self.A @ self.x_hat
        self.x_hat[3] += self.gravity_y * self.dt
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten()

    def update(self, measurement):
        measurement = measurement.reshape(2, 1)
        y = measurement - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def initialize_state(self, measurement, instant_velocity=None):
        self.x_hat.fill(0.)
        self.x_hat[:2] = measurement.reshape(2, 1)
        if instant_velocity is not None:
            self.x_hat[2:] = instant_velocity.reshape(2, 1)
        self.P = np.eye(4) * 100

# --- Outlier Detector from V14.1 ---
class OutlierDetector:
    def __init__(self, position_threshold, velocity_threshold, max_frames):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=max_frames)
        self.velocity_buffer = deque(maxlen=max_frames)

    def add_measurement(self, position, velocity=None):
        self.position_buffer.append(position.copy())
        if velocity is not None:
            self.velocity_buffer.append(velocity.copy())
    
    def is_outlier(self, new_position, new_velocity=None):
        if len(self.position_buffer) < self.position_buffer.maxlen:
            return False
        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold
        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= self.velocity_buffer.maxlen - 1:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold
        return is_position_outlier or is_velocity_outlier

    def reset(self):
        self.position_buffer.clear()
        self.velocity_buffer.clear()

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)])
        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape

        self.ball_model = RFDETRMedium(pretrain_weights=args.ball_model_path)
        self.player_model = RFDETRMedium(pretrain_weights=args.player_model_path)
        self.ball_model.optimize_for_inference()
        self.player_model.optimize_for_inference()
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = PhysicsKalmanFilter(dt=1.0 / args.fps, gravity_y=GRAVITY_PIXELS_PER_FRAME_SQUARED)
        self.outlier_detector = OutlierDetector(OUTLIER_POSITION_THRESHOLD, OUTLIER_VELOCITY_THRESHOLD, OUTLIER_HISTORY_FRAMES)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
        self.prev_position_abs = None
        self.optical_flow_points_rel = None
        self.prev_gray_frame = None
        self.optical_flow_gap_counter = 0
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # V11 Feature: State machine variables
        self.ball_state = BallState.ON_GROUND
        self.on_ground_frames_count = 0

    def _update_ball_state(self, ball_pos_abs, pitch_mask, coord_transform):
        if not self.track_initialized: return
        ball_pos_rel = coord_transform.abs_to_rel(np.array([ball_pos_abs]))[0]
        x, y = int(ball_pos_rel[0]), int(ball_pos_rel[1])
        h, w = pitch_mask.shape
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        is_on_pitch = pitch_mask[y, x] == 255
        vertical_velocity = self.kf.x_hat[3, 0]

        if self.ball_state == BallState.ON_GROUND:
            if not is_on_pitch or vertical_velocity < UPWARD_VELOCITY_THRESHOLD:
                self.ball_state = BallState.IN_AIR
                self.on_ground_frames_count = 0
        elif self.ball_state == BallState.IN_AIR:
            if is_on_pitch and vertical_velocity >= 0:
                self.on_ground_frames_count += 1
            else:
                self.on_ground_frames_count = 0
            if self.on_ground_frames_count >= ON_GROUND_CONFIRMATION_THRESHOLD:
                self.ball_state = BallState.ON_GROUND

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.args.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue
                annotated_frame = frame.copy()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                coord_transform = self.motion_estimator.update(frame)

                # --- V11 Feature: Pitch Detection ---
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
                color_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pitch_mask = np.zeros_like(color_mask)
                if contours:
                    pitch_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(pitch_contour) > (self.frame_width * self.frame_height * 0.1):
                        cv2.drawContours(pitch_mask, [pitch_contour], -1, 255, thickness=cv2.FILLED)

                # --- V11 Feature: Player Detection ---
                player_detections = self.player_model.predict(frame, confidence=PLAYER_CONFIDENCE)
                player_detections = player_detections[np.isin(player_detections.class_id, PLAYER_CLASS_IDS)]

                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                if predicted_pos_abs is not None:
                    self._update_ball_state(predicted_pos_abs, pitch_mask, coord_transform)

                ball_detections = self.ball_model.predict(frame, confidence=BALL_CONFIDENCE)
                ball_detections = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
                
                # --- V11 Feature: Exclusion Zone ---
                if len(ball_detections) > 0 and len(player_detections) > 0:
                    ball_centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    penalized_confidences = ball_detections.confidence.copy()
                    for i, center in enumerate(ball_centers_rel):
                        if is_point_in_boxes(center, player_detections.xyxy):
                            penalized_confidences[i] *= EXCLUSION_CONFIDENCE_PENALTY
                    ball_detections.confidence = penalized_confidences
                
                measurement_abs, annotation_sv, annotation_label = None, None, ""

                # --- V11 Feature: Adaptive Gating ---
                validation_gate = GATE_SIZE_GROUND if self.ball_state == BallState.ON_GROUND else GATE_SIZE_AIR
                
                if len(ball_detections.xyxy) > 0:
                    centers_abs = coord_transform.rel_to_abs(ball_detections.get_anchors_coordinates(sv.Position.CENTER))
                    best_detection_idx = -1
                    if self.track_initialized and predicted_pos_abs is not None:
                        distances = np.linalg.norm(centers_abs - predicted_pos_abs, axis=1)
                        valid_indices = np.where(distances < validation_gate)[0]
                        if len(valid_indices) > 0:
                            best_detection_idx = valid_indices[np.argmax(ball_detections.confidence[valid_indices])]
                    else:
                        best_detection_idx = np.argmax(ball_detections.confidence)

                    if best_detection_idx != -1:
                        current_pos_abs = centers_abs[best_detection_idx]
                        current_vel_abs = (current_pos_abs - self.prev_position_abs) / self.kf.dt if self.prev_position_abs is not None else None
                        
                        # --- V14.1 Feature: Outlier Detector ---
                        if not self.outlier_detector.is_outlier(current_pos_abs, current_vel_abs):
                            measurement_abs = current_pos_abs
                            self.outlier_detector.add_measurement(current_pos_abs, self.kf.x_hat[2:].flatten())
                            self.optical_flow_gap_counter = 0
                            annotation_sv = ball_detections[best_detection_idx:best_detection_idx + 1]
                            annotation_label = f"Ball ({self.ball_state.name})"

                # --- V14.1 Feature: Optical Flow ---
                if measurement_abs is None and self.track_initialized and self.optical_flow_gap_counter < OPTICAL_FLOW_GAP:
                    if self.optical_flow_points_rel is not None and self.prev_gray_frame is not None:
                        new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points_rel, None, **self.lk_params)
                        if status[0][0] == 1:
                            measurement_abs = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                            self.optical_flow_gap_counter += 1
                            annotation_label = f"Ball (OF)"
                            x_rel, y_rel = new_points_rel[0].ravel()
                            synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))

                if measurement_abs is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        # --- V11 Feature: Action Zone (Kick Detection) ---
                        is_kick = self.ball_state == BallState.ON_GROUND and \
                                  is_point_in_action_zones(coord_transform.abs_to_rel(np.array([measurement_abs]))[0], player_detections.xyxy)
                        if is_kick:
                            instant_velocity = (measurement_abs - predicted_pos_abs) / self.kf.dt
                            if instant_velocity[1] < UPWARD_VELOCITY_THRESHOLD:
                                self.ball_state = BallState.IN_AIR
                                self.kf.initialize_state(measurement_abs, instant_velocity=instant_velocity)
                            else: # Not a strong enough upward motion
                                self.kf.update(measurement_abs)
                        else:
                            self.kf.update(measurement_abs)

                    self.prev_position_abs = self.kf.x_hat[:2].flatten()
                    current_pos_rel = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
                    self.optical_flow_points_rel = np.array([[current_pos_rel]], dtype=np.float32)

                    if annotation_sv is not None:
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[annotation_label])
                        box_to_write = annotation_sv.xyxy[0]
                        pred_file.write(f"{frame_count+1},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n")
                else:
                    self.track_initialized = False
                    self.outlier_detector.reset()
                    self.optical_flow_points_rel = None
                    self.prev_position_abs = None

                out_writer.write(annotated_frame)
                self.prev_gray_frame = gray_frame.copy()

        out_writer.release()
        print("Processing complete for V14.2.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V14.2: The Great Merge.")
    parser.add_argument("--ball_model_path", default=BALL_MODEL_PATH, type=str)
    parser.add_argument("--player_model_path", default=PLAYER_MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    processor = VideoProcessor(args)
    processor.run()
