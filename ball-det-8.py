import cv2
import torch
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from collections import deque
import os
import argparse

# --- CONFIGURATION BLOCK ---
# --- Paths ---
MODEL_PATH = "ball.pth"
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v8.mp4"
PREDICTION_FILE_PATH = "predictions_v8.txt"

# --- Model & Tracking Parameters ---
CONFIDENCE = 0.7
BALL_CLASS_ID = 0
FPS = 25

# --- Outlier Detection Parameters (from your script) ---
POSITION_THRESHOLD = 50.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 3

# --- V6: Optical Flow Parameters ---
MAX_OPTICAL_FLOW_GAP = 4 # Max frames to trust optical flow without a new detection

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
        self.Q = np.eye(6) * 0.1
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 100

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
    """Outlier detection logic from your provided script."""
    def __init__(self, position_threshold=50.0, velocity_threshold=100.0, max_frames=3):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=max_frames)
        self.velocity_buffer = deque(maxlen=max_frames)
        self.outlier_frames = 0
        self.outlier_wait_frames = 4

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
        if new_velocity is not None and len(self.velocity_buffer) >= HISTORY_FRAMES -1:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold
        
        is_outlier = is_position_outlier or is_velocity_outlier
        
        if is_outlier:
            self.outlier_frames += 1
        else:
            self.outlier_frames = 0
            
        if self.outlier_frames >= self.outlier_wait_frames:
            self.reset()
            return True
            
        return is_outlier

    def reset(self):
        self.outlier_frames = 0
        self.position_buffer.clear()
        self.velocity_buffer.clear()

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
            raise IOError(f"No images found in directory: {args.image_dir}")

        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape
        self.fps = args.fps

        self.model = RFDETRMedium(pretrain_weights=args.model_path)
        self.model.optimize_for_inference()
        
        # --- FIX: Removed unexpected 'text_color' argument ---
        self.box_annotator = sv.BoxAnnotator(
            thickness=2, 
            color_lookup=sv.ColorLookup.INDEX
        )
        # --- END FIX ---
        
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)

        self.kf = AdaptiveKalmanFilter(dt=1.0 / self.fps)
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        
        self.optical_flow_points = None
        self.prev_gray_frame = None
        self.optical_flow_gap_counter = 0
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.track_initialized = False
        self.prev_position = None
        self.VALIDATION_GATE_THRESHOLD = 50

    def _get_best_detection(self, ball_detections, predicted_pos):
        if len(ball_detections.xyxy) == 0:
            return None
        
        centers = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if self.track_initialized and predicted_pos is not None:
            distances = np.linalg.norm(centers - predicted_pos, axis=1)
            valid_indices = np.where(distances < self.VALIDATION_GATE_THRESHOLD)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[np.argmin(distances[valid_indices])]
                return {"center": centers[idx], "sv": ball_detections[idx:idx + 1]}
        
        best_idx = np.argmax(ball_detections.confidence)
        return {"center": ball_detections.get_anchors_coordinates(sv.Position.CENTER)[best_idx], "sv": ball_detections[best_idx:best_idx+1]}

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.model.predict(frame, confidence=self.args.confidence)
                ball_detections = detections[detections.class_id == BALL_CLASS_ID]
                
                predicted_pos = self.kf.predict() if self.track_initialized else None
                best_detection = self._get_best_detection(ball_detections, predicted_pos)
                
                measurement = None
                annotation_label = ""
                annotation_sv = None
                
                if best_detection:
                    current_pos = best_detection["center"]
                    current_vel = current_pos - self.prev_position if self.prev_position is not None else None
                    
                    if not self.outlier_detector.is_outlier(current_pos, current_vel):
                        measurement = current_pos
                        self.outlier_detector.add_measurement(current_pos, current_vel)
                        self.optical_flow_gap_counter = 0
                        annotation_label = "Ball (Detected)"
                        annotation_sv = best_detection["sv"]
                        self.box_annotator.color = sv.Color.GREEN
                
                if measurement is None and self.track_initialized and self.optical_flow_gap_counter < MAX_OPTICAL_FLOW_GAP:
                    if self.optical_flow_points is not None and self.prev_gray_frame is not None:
                        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points, None, **self.lk_params)
                        if status[0][0] == 1:
                            measurement = new_points[0].ravel()
                            self.optical_flow_gap_counter += 1
                            annotation_label = f"Ball (OF {self.optical_flow_gap_counter})"
                            self.box_annotator.color = sv.Color.YELLOW
                            
                            x, y = measurement
                            box_size = 20
                            synthetic_box = np.array([x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([BALL_CLASS_ID]))

                if measurement is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement)
                        self.track_initialized = True
                    else:
                        self.kf.update(measurement)
                    
                    self.prev_position = measurement.copy()
                    self.optical_flow_points = np.array([[measurement[0], measurement[1]]], dtype=np.float32)

                    box = annotation_sv.xyxy[0]
                    line = f"{frame_count+1},-1,{box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]},1,-1,-1,-1\n"
                    pred_file.write(line)
                else:
                    self.track_initialized = False
                    self.outlier_detector.reset()
                    self.optical_flow_points = None

                if self.track_initialized and annotation_sv:
                    frame = self.box_annotator.annotate(scene=frame, detections=annotation_sv)
                    frame = self.label_annotator.annotate(scene=frame, detections=annotation_sv, labels=[annotation_label])

                out_writer.write(frame)
                self.prev_gray_frame = gray_frame.copy()

        out_writer.release()
        print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid inference and tracking on a directory of images.")
    parser.add_argument("--model_path", default=MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str, help="Path to the directory containing image frames.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS, help="Frame rate for the output video and Kalman filter.")
    args = parser.parse_args()
    
    processor = VideoProcessor(args)
    processor.run()
