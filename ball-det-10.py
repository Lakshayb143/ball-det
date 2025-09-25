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
MODEL_PATH = "ball.pth"
IMAGE_DIR_PATH = "snmot196"
OUTPUT_PATH = "output_v10.mp4"
PREDICTION_FILE_PATH = "predictions_v10.txt"

# --- Model & Tracking Parameters ---
CONFIDENCE = 0.7
BALL_CLASS_ID = 0
FPS = 25

# --- V10: State Machine & Adaptive Logic Parameters ---
# The vertical velocity (negative is up) needed to trigger an IN_AIR state change
UPWARD_VELOCITY_THRESHOLD = -25.0
# The number of consecutive frames the ball must be on the pitch to be considered ON_GROUND again
ON_GROUND_CONFIRMATION_THRESHOLD = 2
# Gate sizes (in absolute pixels) for each state
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 150.0

# --- Outlier Detection Parameters ---
POSITION_THRESHOLD = 50.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 4

# --- V10: State Enumeration ---
class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2

class AdaptiveKalmanFilter:
    """Kalman Filter with Constant Acceleration model. State: [x, y, vx, vy, ax, ay]."""
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
        self.A = np.array([
            [1, 0, dt, 0, dt2, 0], [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ])
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
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
        self.prev_position_abs = None
        
        # --- V10: Initialize state machine ---
        self.ball_state = BallState.ON_GROUND
        self.on_ground_frames_count = 0

    def _update_ball_state(self, ball_pos_abs, pitch_mask, coord_transform):
        """Updates the ball's state (ON_GROUND/IN_AIR) based on position and velocity."""
        if not self.track_initialized:
            return

        # Convert the absolute ball position to relative for checking against the mask
        ball_pos_rel = coord_transform.abs_to_rel(np.array([ball_pos_abs]))[0]
        x, y = int(ball_pos_rel[0]), int(ball_pos_rel[1])
        
        # Ensure coordinates are within the mask boundaries
        h, w = pitch_mask.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        is_on_pitch = pitch_mask[y, x] == 255
        vertical_velocity = self.kf.x_hat[3, 0] # vy from the state vector [x, y, vx, vy, ax, ay]

        if self.ball_state == BallState.ON_GROUND:
            # Condition to switch to IN_AIR: off the pitch OR a strong upward kick
            if not is_on_pitch or vertical_velocity < UPWARD_VELOCITY_THRESHOLD:
                self.ball_state = BallState.IN_AIR
                self.on_ground_frames_count = 0
        
        elif self.ball_state == BallState.IN_AIR:
            # Condition to switch to ON_GROUND: must be on the pitch and not moving up
            if is_on_pitch and vertical_velocity >= 0:
                self.on_ground_frames_count += 1
            else:
                self.on_ground_frames_count = 0 # Reset if it bounces or is still off pitch

            if self.on_ground_frames_count >= ON_GROUND_CONFIRMATION_THRESHOLD:
                self.ball_state = BallState.ON_GROUND

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, FPS, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue

                annotated_frame = frame.copy()
                
                # --- V10: Robust Pitch Detection ---
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                color_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pitch_mask = np.zeros_like(color_mask)
                if contours:
                    pitch_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(pitch_contour) > (self.frame_width * self.frame_height * 0.1):
                        cv2.drawContours(pitch_mask, [pitch_contour], -1, 255, thickness=cv2.FILLED)
                
                coord_transform = self.motion_estimator.update(frame)
                
                # --- 1. PREDICTION ---
                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                
                # --- 2. STATE UPDATE ---
                if predicted_pos_abs is not None:
                    self._update_ball_state(predicted_pos_abs, pitch_mask, coord_transform)
                
                # --- 3. DETECTION & ASSOCIATION with Adaptive Gate ---
                detections = self.model.predict(frame, confidence=self.args.confidence)
                ball_detections = detections[detections.class_id == BALL_CLASS_ID]
                
                # V10: Set validation gate size based on the current state
                validation_gate = GATE_SIZE_GROUND if self.ball_state == BallState.ON_GROUND else GATE_SIZE_AIR
                
                measurement_abs = None
                annotation_sv = None
                
                if len(ball_detections.xyxy) > 0:
                    centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    centers_abs = coord_transform.rel_to_abs(centers_rel)

                    best_detection_idx = -1
                    if self.track_initialized and predicted_pos_abs is not None:
                        distances = np.linalg.norm(centers_abs - predicted_pos_abs, axis=1)
                        valid_indices = np.where(distances < validation_gate)[0]
                        if len(valid_indices) > 0:
                            best_detection_idx = valid_indices[np.argmin(distances[valid_indices])]
                    else:
                        best_detection_idx = np.argmax(ball_detections.confidence)

                    if best_detection_idx != -1:
                        measurement_abs = centers_abs[best_detection_idx]
                        annotation_sv = ball_detections[best_detection_idx:best_detection_idx + 1]

                # --- 4. UPDATE ---
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
        print("Processing complete for V10.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V10: Ball tracking with state machine and adaptive logic.")
    parser.add_argument("--model_path", default=MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    # Corrected VideoWriter initialization in main logic as per user's previous working scripts.
    # The VideoProcessor class uses args directly.
    processor = VideoProcessor(args)
    processor.run()
