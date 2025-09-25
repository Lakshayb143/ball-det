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
BALL_MODEL_PATH = "ball.pth"
PLAYER_MODEL_PATH = "player.pth"
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v15.mp4"
PREDICTION_FILE_PATH = "predictions_v15.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.65 
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 25

# --- V15: EKF Physics Model Parameters ---
# Represents the effect of gravity in pixels per frame squared.
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.2
# A small, unitless value for air resistance. The key tuning parameter for the EKF.
DRAG_COEFFICIENT = 0.01 

# --- V13 Resilience & Tuning Parameters ---
Q_SCALE_DEFAULT = 2.5   
Q_SCALE_COASTING = 15.0  
MAX_COASTING_FRAMES = 18 
MAX_GROUND_LOST_FRAMES = 6 
REACQUISITION_GATE_MULTIPLIER = 2.0 

# --- V11 Context Parameters ---
EXCLUSION_CONFIDENCE_PENALTY = 0.2
ACTION_ZONE_HEIGHT_PERCENT = 0.15
ACTION_ZONE_WIDTH_EXPANSION_PERCENT = 0.1
UPWARD_VELOCITY_THRESHOLD = -25.0
ON_GROUND_CONFIRMATION_THRESHOLD = 2
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 120.0

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    COASTING = 3

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

# --- V15: New Extended Kalman Filter ---
class ExtendedKalmanFilter:
    def __init__(self, dt=1.0, gravity_y=0.0, drag_coeff=0.0):
        self.dt = dt
        self.gravity_y = gravity_y
        self.drag_coeff = drag_coeff
        
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # Measurement matrix
        self.R = np.eye(2) * 5.0 # Measurement noise
        self.x_hat = np.zeros((4, 1)) # State estimate [x, y, vx, vy]
        self.P = np.eye(4) * 100 # Estimate covariance
        self.set_process_noise(Q_SCALE_DEFAULT)

    def set_process_noise(self, q_scale):
        self.Q = np.eye(4) * q_scale

    def predict(self):
        # --- 1. Predict next state using non-linear model f(x) ---
        x, y, vx, vy = self.x_hat.flatten()
        dt = self.dt
        
        v = np.sqrt(vx**2 + vy**2)
        ax_drag = -self.drag_coeff * v * vx
        ay_drag = -self.drag_coeff * v * vy
        
        # Update velocities with drag and gravity
        vx_new = vx + ax_drag * dt
        vy_new = vy + (ay_drag + self.gravity_y) * dt
        
        # Update positions with old velocities
        x_new = x + vx * dt
        y_new = y + vy * dt
        
        self.x_hat = np.array([x_new, y_new, vx_new, vy_new]).reshape(4, 1)

        # --- 2. Calculate the Jacobian Matrix (F) ---
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Avoid division by zero if velocity is zero
        if v > 1e-6:
            # Partial derivatives of drag with respect to vx and vy
            d_ax_dvx = -self.drag_coeff * (vx**2 / v + v)
            d_ax_dvy = -self.drag_coeff * (vx * vy / v)
            d_ay_dvx = d_ax_dvy
            d_ay_dvy = -self.drag_coeff * (vy**2 / v + v)
            
            F[2, 2] = 1 + d_ax_dvx * dt
            F[2, 3] = d_ax_dvy * dt
            F[3, 2] = d_ay_dvx * dt
            F[3, 3] = 1 + d_ay_dvy * dt

        # --- 3. Predict next covariance ---
        self.P = F @ self.P @ F.T + self.Q
        
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

        # V15: Initialize the new ExtendedKalmanFilter
        self.kf = ExtendedKalmanFilter(dt=1.0 / args.fps, gravity_y=GRAVITY_PIXELS_PER_FRAME_SQUARED, drag_coeff=DRAG_COEFFICIENT)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
        self.ball_state = BallState.ON_GROUND
        self.on_ground_frames_count = 0
        self.lost_frames_count = 0
    
    # ... (the _update_ball_state and run methods remain the same as v13_corrected) ...
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
                
                player_detections = self.player_model.predict(frame, confidence=self.args.player_confidence)
                player_detections = player_detections[np.isin(player_detections.class_id, PLAYER_CLASS_IDS)]

                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                if predicted_pos_abs is not None and self.ball_state != BallState.COASTING:
                    self._update_ball_state(predicted_pos_abs, pitch_mask, coord_transform)
                
                ball_detections = self.ball_model.predict(frame, confidence=self.args.ball_confidence)
                ball_detections = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
                
                if len(ball_detections) > 0 and len(player_detections) > 0:
                    ball_centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    penalized_confidences = ball_detections.confidence.copy()
                    for i, center in enumerate(ball_centers_rel):
                        if is_point_in_boxes(center, player_detections.xyxy):
                            penalized_confidences[i] *= EXCLUSION_CONFIDENCE_PENALTY
                    ball_detections.confidence = penalized_confidences

                if self.ball_state == BallState.ON_GROUND:
                    validation_gate = GATE_SIZE_GROUND
                elif self.ball_state == BallState.IN_AIR:
                    validation_gate = GATE_SIZE_AIR
                else: 
                    validation_gate = GATE_SIZE_AIR * REACQUISITION_GATE_MULTIPLIER

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

                if measurement_abs is not None:
                    self.lost_frames_count = 0
                    self.kf.set_process_noise(Q_SCALE_DEFAULT)
                    
                    if self.ball_state == BallState.COASTING:
                         self._update_ball_state(measurement_abs, pitch_mask, coord_transform)

                    instant_velocity = (measurement_abs - predicted_pos_abs) / self.kf.dt if self.track_initialized else None
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        is_kick = self.ball_state == BallState.ON_GROUND and instant_velocity is not None and \
                                  instant_velocity[1] < UPWARD_VELOCITY_THRESHOLD and \
                                  is_point_in_action_zones(coord_transform.abs_to_rel(np.array([measurement_abs]))[0], player_detections.xyxy)
                        if is_kick:
                            self.ball_state = BallState.IN_AIR
                            self.kf.initialize_state(measurement_abs, instant_velocity=instant_velocity)
                        else:
                            self.kf.update(measurement_abs)
                    
                    label = f"Ball ({self.ball_state.name})"
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[label])
                    pred_file.write(f"{frame_count+1},-1,{annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][1]},{annotation_sv.xyxy[0][2]-annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][3]-annotation_sv.xyxy[0][1]},1,-1,-1,-1\n")
                
                elif self.track_initialized:
                    self.lost_frames_count += 1
                    
                    if self.ball_state == BallState.IN_AIR:
                        self.ball_state = BallState.COASTING
                        self.kf.set_process_noise(Q_SCALE_COASTING)
                    
                    timeout = MAX_COASTING_FRAMES if self.ball_state == BallState.COASTING else MAX_GROUND_LOST_FRAMES
                    if self.lost_frames_count > timeout:
                        self.track_initialized = False
                        self.lost_frames_count = 0
                        self.ball_state = BallState.ON_GROUND
                    else: 
                        predicted_pos_rel = coord_transform.abs_to_rel(np.array([predicted_pos_abs]))[0]
                        x, y = predicted_pos_rel.ravel()
                        box_size = 10 
                        synthetic_box = np.array([x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2])
                        synthetic_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id = np.array([0]))
                        label = f"Ball ({self.ball_state.name})"
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=synthetic_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=synthetic_sv, labels=[label])

                out_writer.write(annotated_frame)

        out_writer.release()
        print("Processing complete for V15.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V15: Ball tracking with an Extended Kalman Filter (EKF).")
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
