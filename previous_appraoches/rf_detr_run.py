import cv2
import torch
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque
import os

# --- CONFIGURATION BLOCK ---
# --- Paths ---
MODEL_PATH = "ball.pth"
IMAGE_DIR_PATH = "snmot200"  # MODIFIED: Path to the directory of images
OUTPUT_PATH = "output_rfrun_snmot200.mp4"
PREDICTION_FILE_PATH = "predictions_rfrun_snmot200.txt" # NEW: Name for the output prediction file

# --- Model & Tracking Parameters ---
CONFIDENCE = 0.8
BALL_CLASS_ID = 0
FPS = 25  # MODIFIED: Set the desired FPS for the output video and Kalman Filter

# --- Outlier Detection Parameters ---
POSITION_THRESHOLD = 50.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 3


class AdaptiveKalmanFilter:
    """Kalman Filter with Constant Acceleration model. State: [x, y, vx, vy, ax, ay]."""

    def __init__(self, dt=1.0):
        self.dt = dt
        dt2 = 0.5 * dt ** 2
        self.A = np.array([
            [1, 0, dt, 0, dt2, 0], [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.Q = np.eye(6)
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 100
        self.set_process_noise(10.0)

    def set_process_noise(self, accel_noise):
        self.Q[4, 4] = self.Q[5, 5] = accel_noise

    def predict(self):
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

class OutlierDetector:
    """Outlier detection with 4-frame confirmation logic."""
    
    def __init__(self, position_threshold=50.0, velocity_threshold=100.0, max_frames=3):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=max_frames)
        self.velocity_buffer = deque(maxlen=max_frames)
        self.outlier_frames = 0
        self.outlier_wait_frames = 4
        self.tracking_suspended = False
        self.suspension_frames = 0
        self.max_suspension_frames = 3
        
    def add_frame(self, position, velocity=None):
        self.position_buffer.append(position.copy())
        if velocity is not None:
            self.velocity_buffer.append(velocity.copy())
    
    def is_outlier(self, new_position, new_velocity=None):
        if len(self.position_buffer) < 3:
            return False, True
        
        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold
        
        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= 2:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold
        
        is_outlier = is_position_outlier or is_velocity_outlier
        
        if is_outlier:
            self.outlier_frames += 1
            if self.outlier_frames >= self.outlier_wait_frames:
                self._reset_tracking()
                return True, False
            else:
                self.tracking_suspended = True
                self.suspension_frames = self.outlier_frames
                return True, False
        else:
            self.outlier_frames = 0
            self.tracking_suspended = False
            self.suspension_frames = 0
            return False, True
    
    def _reset_tracking(self):
        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0
        self.position_buffer.clear()
        self.velocity_buffer.clear()
    
    def should_reset_tracking(self):
        if self.tracking_suspended and self.suspension_frames >= self.max_suspension_frames:
                self._reset_tracking()
                return True
        return False

class VideoProcessor:
    def __init__(self):
        # --- MODIFIED: Initialize from image directory ---
        self.image_files = sorted([os.path.join(IMAGE_DIR_PATH, f) for f in os.listdir(IMAGE_DIR_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
            raise IOError(f"No images found in directory: {IMAGE_DIR_PATH}")

        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape
        self.fps = FPS
        # --- END MODIFICATION ---

        self.model = RFDETRMedium(pretrain_weights=MODEL_PATH)
        self.model.optimize_for_inference()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = AdaptiveKalmanFilter(dt=1.0 / self.fps)
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        self.track_lost_count = self.track_hit_streak = 0
        self.max_lost_frames = int(self.fps * 0.75)
        self.STABLE_TRACK_THRESHOLD = 5
        self.VALIDATION_GATE_THRESHOLD = 25
        self.prev_position = None

    def _get_best_detection(self, ball_detections, predicted_pos, track_initialized):
        if len(ball_detections.xyxy) == 0:
            return None
        
        high_conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(high_conf_mask):
            return None
        
        filtered_detections = ball_detections[high_conf_mask]
        centers = filtered_detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if track_initialized and predicted_pos is not None:
            distances = np.linalg.norm(centers - predicted_pos, axis=1)
            valid_indices = np.where(distances < self.VALIDATION_GATE_THRESHOLD)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[np.argmin(distances[valid_indices])]
            else:
                idx = np.argmax(filtered_detections.confidence)
        else:
            idx = np.argmax(filtered_detections.confidence)
        
        return {"center": centers[idx], "sv": filtered_detections[idx:idx + 1]}

    def _process_detection(self, best_detection, track_initialized, frame_count):
        if not best_detection:
            return track_initialized, False
        
        current_position = best_detection["center"]
        current_velocity = current_position - self.prev_position if self.prev_position is not None else None
        is_outlier, should_use_detection = self.outlier_detector.is_outlier(current_position, current_velocity)
        
        if should_use_detection:
            self.track_hit_streak += 1
            self.track_lost_count = 0
            self.kf.set_process_noise(1.0 if self.track_hit_streak > self.STABLE_TRACK_THRESHOLD else 10.0)
            
            if not track_initialized:
                self.kf.initialize_state(current_position)
                track_initialized = True
            else:
                self.kf.update(current_position)
            
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position = current_position.copy()
            print(f"Frame {frame_count+1}: ACCEPTED detection at {current_position}")
        else:
            self.track_lost_count += 1
            self.track_hit_streak = 0
            self.kf.set_process_noise(10.0)
            print(f"Frame {frame_count+1}: OUTLIER detected, rejecting prediction at {current_position}")
            
            if self.outlier_detector.should_reset_tracking():
                track_initialized = False
                self.track_lost_count = 0
                self.track_hit_streak = 0
                print(f"Frame {frame_count+1}: Resetting tracking")
        
        return track_initialized, should_use_detection

    def _annotate_frame(self, annotated_frame, track_initialized, best_detection, should_use_detection):
        if best_detection and should_use_detection and track_initialized and self.track_lost_count < self.max_lost_frames:
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=best_detection["sv"])
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=best_detection["sv"], labels=["Ball"])

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, self.fps, (self.frame_width, self.frame_height))
        track_initialized = False
        
        # --- MODIFIED: Open prediction file with 'with' statement ---
        with open(PREDICTION_FILE_PATH, "w") as pred_file:
            # --- MODIFIED: Loop through image files instead of video frames ---
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Warning: Could not read frame {image_path}")
                    continue
                
                detections = self.model.predict(frame, confidence=CONFIDENCE)
                annotated_frame = frame.copy()

                ball_detections = detections[detections.class_id == BALL_CLASS_ID]
                predicted_pos = self.kf.predict() if track_initialized else None
                
                best_detection = self._get_best_detection(ball_detections, predicted_pos, track_initialized)
                track_initialized, should_use_detection = self._process_detection(best_detection, track_initialized, frame_count)
                
                # --- NEW: Write accepted detections to prediction file ---
                if best_detection and should_use_detection:
                    box = best_detection["sv"].xyxy[0]
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                    line = f"{frame_count+1},-1,{x1},{y1},{w},{h},1,-1,-1,-1\n"
                    pred_file.write(line)
                # --- END NEW ---

                if not best_detection and track_initialized:
                    self.track_lost_count += 1
                    self.track_hit_streak = 0
                    self.kf.set_process_noise(10.0)

                self._annotate_frame(annotated_frame, track_initialized, best_detection, should_use_detection)

                if self.track_lost_count >= self.max_lost_frames and track_initialized:
                    track_initialized = self.track_hit_streak = self.track_lost_count = 0

                out_writer.write(annotated_frame)

        out_writer.release()
        print("Processing complete.")

# --- Main Execution ---
if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()