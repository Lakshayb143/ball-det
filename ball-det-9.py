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
OUTPUT_PATH = "output_v9.mp4"
PREDICTION_FILE_PATH = "predictions_v9.txt"

# --- Model & Tracking Parameters ---
CONFIDENCE = 0.7
BALL_CLASS_ID = 0
FPS = 25

# --- V9: Velocity Gate Parameters ---
HISTORY_LENGTH = 5 # Number of past detections to calculate average velocity
VELOCITY_GATE_BUFFER = 30 # An extra buffer in pixels to add to the gate

class AdaptiveKalmanFilter:
    """Kalman Filter with Constant Acceleration model. Used here for interpolation."""
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

class BallTrackerV9:
    """
    V9: A 'detection-first' tracker. Uses a velocity-based outlier gate on
    detector outputs and falls back to Kalman Filter for interpolation.
    """
    def __init__(self, dt=1.0/25.0):
        self.kf = AdaptiveKalmanFilter(dt=dt)
        self.history = deque(maxlen=HISTORY_LENGTH) # Stores (frame_count, position)
        self.last_known_sv = None
        self.lost_counter = 0
        self.MAX_LOST_FRAMES = int(FPS * 0.5) # Allow coasting for half a second

    def _calculate_average_velocity(self):
        if len(self.history) < 2:
            return 30.0 # Return a reasonable default
        
        velocities = []
        history_list = list(self.history)
        for i in range(len(history_list) - 1):
            frame1, pos1 = history_list[i]
            frame2, pos2 = history_list[i+1]
            delta_frames = frame2 - frame1
            if delta_frames > 0:
                distance = np.linalg.norm(pos2 - pos1)
                velocities.append(distance / delta_frames)
        
        return np.mean(velocities) if velocities else 30.0

    def update(self, detections, frame_count):
        # Step 1: Find the single best detection candidate based on confidence
        best_candidate = None
        if len(detections.xyxy) > 0:
            best_idx = np.argmax(detections.confidence)
            best_candidate = {
                "sv": detections[best_idx:best_idx+1],
                "center": detections.get_anchors_coordinates(sv.Position.CENTER)[best_idx]
            }
        
        # Step 2: The "Velocity Gate" Outlier Test
        is_valid_detection = False
        if best_candidate:
            if not self.history: # This is the first ever detection
                is_valid_detection = True
            else:
                last_frame, last_pos = self.history[-1]
                delta_frames = frame_count - last_frame
                avg_velocity = self._calculate_average_velocity()
                
                max_allowed_distance = (avg_velocity * delta_frames) + VELOCITY_GATE_BUFFER
                actual_distance = np.linalg.norm(best_candidate["center"] - last_pos)
                
                if actual_distance <= max_allowed_distance:
                    is_valid_detection = True

        # Step 3: Decide whether to use the valid detection or interpolate
        final_sv_for_annotation = None
        annotation_label = ""
        box_color = sv.Color.RED # Default to red for interpolated/lost

        if is_valid_detection:
            # --- PATH A: Valid Detection Found ---
            self.lost_counter = 0
            
            # Update the Kalman Filter with the real measurement
            self.kf.update(best_candidate["center"]) 
            
            # Update our history buffer with this verified detection
            self.history.append((frame_count, best_candidate["center"]))
            self.last_known_sv = best_candidate["sv"]
            
            final_sv_for_annotation = self.last_known_sv
            annotation_label = "Ball (Detected)"
            box_color = sv.Color.GREEN

        else:
            # --- PATH B: No Valid Detection - Interpolate ---
            self.lost_counter += 1
            
            # If we have no history or have been lost for too long, we give up
            if not self.history or self.lost_counter > self.MAX_LOST_FRAMES:
                self.history.clear()
                self.last_known_sv = None
                return None, "", None
            
            # Use the Kalman Filter's prediction as the interpolated point
            interpolated_pos = self.kf.predict()
            
            # Create a synthetic box for annotation using the last known box size
            last_box = self.last_known_sv.xyxy[0]
            w, h = last_box[2] - last_box[0], last_box[3] - last_box[1]
            x, y = interpolated_pos
            synthetic_box = np.array([x - w/2, y - h/2, x + w/2, y + h/2])
            final_sv_for_annotation = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([BALL_CLASS_ID]))
            
            annotation_label = f"Ball (Interpolated {self.lost_counter})"
            box_color = sv.Color.YELLOW

        return final_sv_for_annotation, annotation_label, box_color


class ImageProcessor:
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
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.ball_tracker = BallTrackerV9(dt=1.0 / self.fps)

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files):
                frame = cv2.imread(image_path)
                if frame is None: continue
                
                detections = self.model.predict(frame, confidence=self.args.confidence)
                ball_detections = detections[detections.class_id == BALL_CLASS_ID]
                
                final_sv, label, color = self.ball_tracker.update(ball_detections, frame_count)
                
                if final_sv:
                    # Write prediction to file
                    box = final_sv.xyxy[0]
                    line = f"{frame_count+1},-1,{box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]},1,-1,-1,-1\n"
                    pred_file.write(line)
                    
                    # Annotate frame
                    self.box_annotator.color = color
                    frame = self.box_annotator.annotate(scene=frame, detections=final_sv)
                    frame = self.label_annotator.annotate(scene=frame, detections=final_sv, labels=[label])

                out_writer.write(frame)

        out_writer.release()
        print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V9 detection-first tracking on a directory of images.")
    parser.add_argument("--model_path", default=MODEL_PATH, type=str)
    parser.add_argument("--image_dir", default=IMAGE_DIR_PATH, type=str)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--prediction_path", type=str, default=PREDICTION_FILE_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    processor = ImageProcessor(args)
    processor.run()
