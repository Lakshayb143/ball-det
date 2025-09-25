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
OUTPUT_PATH = "output_v9_4.mp4"
PREDICTION_FILE_PATH = "predictions_v9_4.txt"

# --- Model & Tracking Parameters ---
CONFIDENCE = 0.7
BALL_CLASS_ID = 0
FPS = 25
BALL_TRACKER_BUFFER_SIZE = 3# Buffer size for the new BallTracker

# --- Optical Flow Parameters ---
MAX_OPTICAL_FLOW_GAP = 3

# --- V9_4: New Centroid-Based BallTracker ---
class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        # If no detections, return empty and don't pollute buffer
        if len(detections) == 0:
            return sv.Detections.empty()

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        # Need at least one frame in buffer to calculate centroid
        if not self.buffer:
            return sv.Detections.empty()

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]
    
    def reset(self):
        self.buffer.clear()

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

        self.kf = AdaptiveKalmanFilter(dt=1.0 / args.fps)
        self.ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)
        
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
                
                # --- V9_4: Use the new BallTracker to select the best detection ---
                best_detection_sv = self.ball_tracker.update(ball_detections)
                
                measurement_abs = None
                annotation_sv = best_detection_sv
                annotation_label = ""
                
                if len(best_detection_sv) > 0:
                    center_rel = best_detection_sv.get_anchors_coordinates(sv.Position.CENTER)
                    measurement_abs = coord_transform.rel_to_abs(center_rel).flatten()
                    annotation_label = "Ball (Detected)"
                    self.optical_flow_gap_counter = 0

                # If no detection, try to find a measurement using Optical Flow
                if measurement_abs is None and self.track_initialized and self.optical_flow_gap_counter < MAX_OPTICAL_FLOW_GAP:
                    if self.optical_flow_points_rel is not None and self.prev_gray_frame is not None:
                        new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points_rel, None, **self.lk_params)
                        if status[0][0] == 1:
                            measurement_abs = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                            self.optical_flow_gap_counter += 1
                            annotation_label = f"Ball (OF)"
                            x_rel, y_rel = new_points_rel[0].ravel()
                            box_size = 20
                            synthetic_box = np.array([x_rel - box_size/2, y_rel - box_size/2, x_rel + box_size/2, y_rel + box_size/2])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))

                if measurement_abs is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        # Even if OF provides a point, we use the KF's smoother prediction to update the final position
                        # unless the measurement is very close to the prediction
                        if predicted_pos_abs is not None and np.linalg.norm(measurement_abs - predicted_pos_abs) < 50:
                             self.kf.update(measurement_abs)
                        # else, we trust the KF prediction more than a potentially noisy OF point
                        
                    self.prev_position_abs = self.kf.x_hat[:2].flatten()
                    current_pos_rel = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
                    self.optical_flow_points_rel = np.array([[current_pos_rel]], dtype=np.float32)

                    if len(annotation_sv) > 0:
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[annotation_label])
                        box_to_write = annotation_sv.xyxy[0]
                        line = f"{frame_count+1},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n"
                        pred_file.write(line)
                else:
                    self.track_initialized = False
                    self.ball_tracker.reset()
                    self.optical_flow_points_rel = None
                    self.prev_position_abs = None

                out_writer.write(annotated_frame)
                self.prev_gray_frame = gray_frame.copy()

        out_writer.release()
        print("Processing complete for V9_4.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V9_4: Centroid-based ball tracking experiment.")
    parser.add_argument("--model_path", default=MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    processor = VideoProcessor(args)
    processor.run()
