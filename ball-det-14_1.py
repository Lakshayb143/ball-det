import cv2
import torch
import argparse
import os
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator

# --- CONFIGURATION BLOCK ---
MODEL_PATH = "ball.pth"
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v14_1.mp4"
PREDICTION_FILE_PATH = "predictions_v14_1.txt"

# --- Model & Tracking Parameters (from your script) ---
CONFIDENCE = 0.7
BALL_CLASS_ID = 0
FPS = 25
VALIDATION_GATE_THRESHOLD = 45.0 

# --- V14.1: New Physics Model Parameter ---
# Represents the effect of gravity in pixels per frame squared.
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.57

# --- Outlier Detection Parameters (from your script) ---
POSITION_THRESHOLD = 40.0
VELOCITY_THRESHOLD = 40.0
HISTORY_FRAMES = 4

# --- Optical Flow Parameters (from your script) ---
MAX_OPTICAL_FLOW_GAP = 3

# --- V14.1: Replaced Kalman Filter ---
class PhysicsKalmanFilter:
    """
    A Kalman Filter with a 4D state [x, y, vx, vy].
    It incorporates a constant gravity term in its prediction step.
    """
    def __init__(self, dt=1.0, gravity_y=0.0):
        self.dt = dt
        self.gravity_y = gravity_y # Downward acceleration
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.5 # Process noise
        self.R = np.eye(2) * 5.0 # Measurement noise
        self.x_hat = np.zeros((4, 1)) # State estimate [x, y, vx, vy]
        self.P = np.eye(4) * 100 # Estimate covariance

    def predict(self):
        self.x_hat = self.A @ self.x_hat
        # Add the effect of gravity on the vertical velocity
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

    def initialize_state(self, measurement):
        self.x_hat.fill(0.)
        self.x_hat[:2] = measurement.reshape(2, 1)
        self.P = np.eye(4) * 100

class OutlierDetector:
    """Outlier detection logic from your script."""
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
        if len(self.position_buffer) < HISTORY_FRAMES:
            return False
        
        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold
        
        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= HISTORY_FRAMES - 1:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold
        
        return is_position_outlier or is_velocity_outlier

    def reset(self):
        self.position_buffer.clear()
        self.velocity_buffer.clear()

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files: raise IOError(f"No images found in directory: {args.image_dir}")

        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape

        self.model = RFDETRMedium(pretrain_weights=args.model_path)
        self.model.optimize_for_inference()
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        # --- V14.1: Initialize the new PhysicsKalmanFilter ---
        self.kf = PhysicsKalmanFilter(dt=1.0 / args.fps, gravity_y=GRAVITY_PIXELS_PER_FRAME_SQUARED)
        
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
        self.prev_position_abs = None
        
        self.optical_flow_points_rel = None
        self.prev_gray_frame = None
        self.optical_flow_gap_counter = 0
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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
                
                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                
                detections = self.model.predict(frame, confidence=self.args.confidence)
                ball_detections = detections[detections.class_id == BALL_CLASS_ID]
                
                measurement_abs = None
                annotation_sv = None
                annotation_label = ""

                if len(ball_detections.xyxy) > 0:
                    centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    centers_abs = coord_transform.rel_to_abs(centers_rel)

                    best_detection_idx = -1
                    if self.track_initialized and predicted_pos_abs is not None:
                        distances = np.linalg.norm(centers_abs - predicted_pos_abs, axis=1)
                        valid_indices = np.where(distances < VALIDATION_GATE_THRESHOLD)[0]
                        if len(valid_indices) > 0:
                            best_detection_idx = valid_indices[np.argmin(distances[valid_indices])]
                    else:
                        best_detection_idx = np.argmax(ball_detections.confidence)

                    if best_detection_idx != -1:
                        current_pos_abs = centers_abs[best_detection_idx]
                        # V14.1 Note: The new KF only has vx, vy. The outlier detector is compatible.
                        current_vel_abs = self.kf.x_hat[2:].flatten() if self.track_initialized else None
                        
                        if not self.outlier_detector.is_outlier(current_pos_abs, current_vel_abs):
                            measurement_abs = current_pos_abs
                            # V14.1 Note: We add the KF's stable velocity, not a noisy instant velocity.
                            self.outlier_detector.add_measurement(current_pos_abs, current_vel_abs)
                            self.optical_flow_gap_counter = 0
                            annotation_sv = ball_detections[best_detection_idx:best_detection_idx + 1]
                            annotation_label = "Ball (Detected)"

                if measurement_abs is None and self.track_initialized and self.optical_flow_gap_counter < MAX_OPTICAL_FLOW_GAP:
                    if self.optical_flow_points_rel is not None and self.prev_gray_frame is not None:
                        new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points_rel, None, **self.lk_params)
                        if status[0][0] == 1:
                            measurement_abs = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                            self.optical_flow_gap_counter += 1
                            annotation_label = f"Ball (OF {self.optical_flow_gap_counter})"
                            
                            x_rel, y_rel = new_points_rel[0].ravel()
                            box_size = 20
                            synthetic_box = np.array([x_rel - box_size/2, y_rel - box_size/2, x_rel + box_size/2, y_rel + box_size/2])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id = np.array([0]))

                if measurement_abs is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        self.kf.update(measurement_abs)

                    self.prev_position_abs = self.kf.x_hat[:2].flatten()
                    
                    current_pos_rel = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
                    self.optical_flow_points_rel = np.array([[current_pos_rel]], dtype=np.float32)

                    if annotation_sv is not None:
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[annotation_label])
                        
                        box_to_write = annotation_sv.xyxy[0]
                        line = f"{frame_count+1},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n"
                        pred_file.write(line)
                else:
                    self.track_initialized = False
                    self.outlier_detector.reset()
                    self.optical_flow_points_rel = None
                    self.prev_position_abs = None

                out_writer.write(annotated_frame)
                self.prev_gray_frame = gray_frame.copy()

        out_writer.release()
        print("Processing complete for V14.1.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V14.1: Hybrid tracking with Physics-Aware KF.")
    parser.add_argument("--model_path", default=MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    processor = VideoProcessor(args)
    processor.run()
