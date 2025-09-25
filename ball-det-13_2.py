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
OUTPUT_PATH = "output_v13_2.mp4"
PREDICTION_FILE_PATH = "predictions_v13_2.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.65 
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 25

# --- V13.2: New Outlier Detector Parameters ---
POSITION_THRESHOLD = 55.0  # Increased threshold to be a bit more lenient initially
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 3
OUTLIER_WAIT_FRAMES = 4 # How many consecutive outliers trigger a hard reset

# --- V13 Resilience & Tuning Parameters ---
Q_SCALE_DEFAULT = 0.5   
Q_SCALE_COASTING = 11.0  
MAX_COASTING_FRAMES = 18 
MAX_GROUND_LOST_FRAMES = 6 

# --- V12 Physics Model Parameters ---
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.57

# --- V11 Context Parameters ---
EXCLUSION_CONFIDENCE_PENALTY = 0.2
UPWARD_VELOCITY_THRESHOLD = -25.0
ON_GROUND_CONFIRMATION_THRESHOLD = 2

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    COASTING = 3

# --- Helper Functions ---
def is_point_in_boxes(point, boxes):
    return np.any((point[0] >= boxes[:, 0]) & (point[0] <= boxes[:, 2]) & (point[1] >= boxes[:, 1]) & (point[1] <= boxes[:, 3]))

# --- V13.2: New OutlierDetector Class ---
class OutlierDetector:
    def __init__(self, position_threshold, velocity_threshold, max_history_frames, outlier_wait_frames):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=max_history_frames)
        self.velocity_buffer = deque(maxlen=max_history_frames)
        self.outlier_frames_count = 0
        self.outlier_wait_frames = outlier_wait_frames

    def check(self, new_position, new_velocity):
        if len(self.position_buffer) < self.position_buffer.maxlen:
            return True # Not enough history to make a decision, so accept

        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold
        
        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= self.velocity_buffer.maxlen -1:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold
        
        is_outlier = is_position_outlier or is_velocity_outlier
        
        if is_outlier:
            self.outlier_frames_count += 1
            if self.outlier_frames_count >= self.outlier_wait_frames:
                self.reset() # Hard reset after 4 consecutive outliers
                return False # Reject and force re-initialization
            return False # Reject but maintain state (suspension)
        else:
            self.outlier_frames_count = 0
            return True # Accept the detection

    def add_measurement(self, position, velocity):
        self.position_buffer.append(position)
        if velocity is not None:
            self.velocity_buffer.append(velocity)
    
    def reset(self):
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.outlier_frames_count = 0

class PhysicsKalmanFilter:
    def __init__(self, dt=1.0, gravity_y=0.0):
        self.dt = dt
        self.gravity_y = gravity_y
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((4, 1))
        self.P = np.eye(4) * 100
        self.set_process_noise(Q_SCALE_DEFAULT) 

    def set_process_noise(self, q_scale):
        self.Q = np.eye(4) * q_scale

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
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        # V13.2: Initialize the new outlier detector
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES, OUTLIER_WAIT_FRAMES)
        
        self.track_initialized = False
        self.ball_state = BallState.ON_GROUND
        self.lost_frames_count = 0

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.args.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue
                annotated_frame = frame.copy()
                
                coord_transform = self.motion_estimator.update(frame)
                
                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                
                ball_detections = self.ball_model.predict(frame, confidence=self.args.ball_confidence)
                ball_detections = ball_detections[ball_detections.class_id == BALL_CLASS_ID]

                # --- Find best candidate before outlier check ---
                measurement_abs, annotation_sv = None, None
                if len(ball_detections) > 0:
                    centers_abs = coord_transform.rel_to_abs(ball_detections.get_anchors_coordinates(sv.Position.CENTER))
                    best_detection_idx = np.argmax(ball_detections.confidence)
                    measurement_abs = centers_abs[best_detection_idx]
                    annotation_sv = ball_detections[best_detection_idx:best_detection_idx + 1]

                # --- V13.2: New Outlier Detection Logic ---
                should_use_detection = False
                if self.track_initialized and measurement_abs is not None:
                    instant_velocity = (measurement_abs - predicted_pos_abs) / self.kf.dt
                    should_use_detection = self.outlier_detector.check(measurement_abs, instant_velocity)
                    if should_use_detection:
                        self.outlier_detector.add_measurement(measurement_abs, instant_velocity)
                elif not self.track_initialized and measurement_abs is not None:
                    # If track isn't active, we accept the first high-confidence detection
                    should_use_detection = True

                if not should_use_detection:
                    measurement_abs = None # Nullify the measurement if it's an outlier
                    if self.outlier_detector.outlier_frames_count >= OUTLIER_WAIT_FRAMES:
                        self.track_initialized = False # Force hard reset

                # --- Standard Update Logic (using the vetted measurement_abs) ---
                if measurement_abs is not None:
                    self.lost_frames_count = 0
                    self.kf.set_process_noise(Q_SCALE_DEFAULT)
                    
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                        self.outlier_detector.reset()
                        self.outlier_detector.add_measurement(measurement_abs, np.array([0,0]))
                    else:
                        self.kf.update(measurement_abs)
                    
                    label = f"Ball ({self.ball_state.name})" # State machine logic removed for simplicity, focus on outlier
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[label])
                    pred_file.write(f"{frame_count+1},-1,{annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][1]},{annotation_sv.xyxy[0][2]-annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][3]-annotation_sv.xyxy[0][1]},1,-1,-1,-1\n")
                
                elif self.track_initialized:
                    self.lost_frames_count += 1
                    if self.lost_frames_count > MAX_COASTING_FRAMES: # Generic lost counter
                        self.track_initialized = False
                        self.outlier_detector.reset()
                    else:
                        predicted_pos_rel = coord_transform.abs_to_rel(np.array([predicted_pos_abs]))[0]
                        x, y = predicted_pos_rel.ravel()
                        box_size = 10 
                        synthetic_box = np.array([x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2])
                        synthetic_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id = np.array([0]))
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=synthetic_sv)

                out_writer.write(annotated_frame)

        out_writer.release()
        print("Processing complete for V13.2.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V13.2: Ball tracking with advanced outlier detection.")
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
