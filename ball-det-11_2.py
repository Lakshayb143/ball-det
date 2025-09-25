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
PLAYER_MODEL_PATH = "player.pth"
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v11_2_corrected.mp4"
PREDICTION_FILE_PATH = "predictions_v11_2_corrected.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.7
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] # Correct: 1: goalkeeper, 2: player, 3: referee
FPS = 25

# --- V11.1: Exclusion Zone Parameters ---
EXCLUSION_CONFIDENCE_PENALTY = 0.2

# --- V11.2: Action Zone Parameters ---
ACTION_ZONE_HEIGHT_PERCENT = 0.15
ACTION_ZONE_WIDTH_EXPANSION_PERCENT = 0.1

# --- State Machine & Adaptive Logic Parameters ---
UPWARD_VELOCITY_THRESHOLD = -25.0
ON_GROUND_CONFIRMATION_THRESHOLD = 2
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 150.0

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2

# --- Helper Functions ---
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

class AdaptiveKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.A = np.zeros((6, 6))
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.Q = np.eye(6) * 0.1
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 100

    def predict(self):
        dt, dt2 = self.dt, 0.5 * self.dt**2
        self.A = np.array([[1, 0, dt, 0, dt2, 0], [0, 1, 0, dt, 0, dt2],[0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[:2].flatten()

    def update(self, measurement):
        measurement = measurement.reshape(2, 1)
        y = measurement - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def initialize_state(self, measurement):
        self.x_hat.fill(0.)
        self.x_hat[:2] = measurement.reshape(2, 1)
        self.P = np.eye(6) * 100

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files: raise IOError(f"No images found in directory: {args.image_dir}")

        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape

        self.ball_model = RFDETRMedium(pretrain_weights=args.ball_model_path)
        self.ball_model.optimize_for_inference()
        self.player_model = RFDETRMedium(pretrain_weights=args.player_model_path)
        self.player_model.optimize_for_inference()
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = AdaptiveKalmanFilter(dt=1.0 / args.fps)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
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
        out_writer = cv2.VideoWriter(self.args.output_path,fourcc, self.args.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue

                annotated_frame = frame.copy()
                
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
                color_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pitch_mask = np.zeros_like(color_mask)
                if contours:
                    pitch_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(pitch_contour) > (self.frame_width * self.frame_height * 0.1):
                        cv2.drawContours(pitch_mask, [pitch_contour], -1, 255, thickness=cv2.FILLED)
                
                coord_transform = self.motion_estimator.update(frame)
                
                player_detections_raw = self.player_model.predict(frame, confidence=self.args.player_confidence)
                player_detections = player_detections_raw[np.isin(player_detections_raw.class_id, PLAYER_CLASS_IDS)]

                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                if predicted_pos_abs is not None:
                    self._update_ball_state(predicted_pos_abs, pitch_mask, coord_transform)
                
                ball_detections_raw = self.ball_model.predict(frame, confidence=self.args.ball_confidence)
                ball_detections = ball_detections_raw[ball_detections_raw.class_id == BALL_CLASS_ID]
                
                if len(ball_detections) > 0 and len(player_detections) > 0:
                    ball_centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    penalized_confidences = ball_detections.confidence.copy()
                    for i, center in enumerate(ball_centers_rel):
                        if is_point_in_boxes(center, player_detections.xyxy):
                            penalized_confidences[i] *= EXCLUSION_CONFIDENCE_PENALTY
                    ball_detections.confidence = penalized_confidences

                validation_gate = GATE_SIZE_GROUND if self.ball_state == BallState.ON_GROUND else GATE_SIZE_AIR
                measurement_abs, annotation_sv = None, None
                
                if len(ball_detections) > 0:
                    centers_abs = coord_transform.rel_to_abs(ball_detections.get_anchors_coordinates(sv.Position.CENTER))
                    best_detection_idx = -1
                    if self.track_initialized and predicted_pos_abs is not None:
                        distances = np.linalg.norm(centers_abs - predicted_pos_abs, axis=1)
                        valid_indices = np.where(distances < validation_gate)[0]
                        if len(valid_indices) > 0:
                            best_valid_idx_local = np.argmax(ball_detections.confidence[valid_indices])
                            best_detection_idx = valid_indices[best_valid_idx_local]
                    else:
                        best_detection_idx = np.argmax(ball_detections.confidence)

                    if best_detection_idx != -1:
                        measurement_abs = centers_abs[best_detection_idx]
                        annotation_sv = ball_detections[best_detection_idx:best_detection_idx + 1]

                # --- V11.2 (Corrected): Immediate Kick Detection Logic ---
                if self.track_initialized and self.ball_state == BallState.ON_GROUND and measurement_abs is not None:
                    ball_pos_rel = coord_transform.abs_to_rel(np.array([measurement_abs]))[0]
                    if is_point_in_action_zones(ball_pos_rel, player_detections.xyxy):
                        # Calculate instant velocity based on the new measurement
                        dt = 1.0 / self.args.fps
                        instant_velocity = (measurement_abs - predicted_pos_abs) / dt
                        instant_vy = instant_velocity[1]
                        
                        # If a strong upward velocity is detected, force state change
                        if instant_vy < UPWARD_VELOCITY_THRESHOLD:
                            self.ball_state = BallState.IN_AIR
                            self.on_ground_frames_count = 0
                # --- End Corrected Logic ---

                if measurement_abs is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        self.kf.update(measurement_abs)

                    if annotation_sv is not None:
                        label = f"Ball ({self.ball_state.name})"
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[label])
                        
                        box_to_write = annotation_sv.xyxy[0]
                        line = f"{frame_count+1},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n"
                        pred_file.write(line)
                else:
                    self.track_initialized = False

                out_writer.write(annotated_frame)

        out_writer.release()
        print("Processing complete for V11.2 (Corrected).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11.2 (Corrected): Ball tracking with player exclusion and action zones.")
    parser.add_argument("--ball_model_path", default=BALL_MODEL_PATH, type=str)
    parser.add_argument("--player_model_path", default=PLAYER_MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--ball_confidence", type=float, default=BALL_CONFIDENCE)
    parser.add_argument("--player_confidence", type=float, default=PLAYER_CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    processor = VideoProcessor(args)
    processor.run()
