import cv2
import torch
import argparse
import os
import time
import threading
import queue
import json
import numpy as np
import supervision as sv
from loguru import logger
from rfdetr import RFDETRMedium
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from enum import Enum
from collections import deque

# --- Configuration and Setup ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "hybrid_tracker_v4_2.log")
logger.add(log_file_path, rotation="10 MB", level="INFO")


# --- State Enumerations for Hybrid Logic ---
class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2

class AdaptiveVelocityTracker:
    """
    V4_2 Tracker: A simplified architecture that removes the SEARCHING/CONFIRMED
    states. It is always either actively tracking or immediately re-acquiring.
    """
    def __init__(self):
        self.ball_state = BallState.ON_GROUND
        self.confirmed_track = None # This is now the only state variable for the track
        self.detection_history = deque(maxlen=10)
        
        # --- State Machine Parameters from your last version ---
        self.on_ground_frames = 0
        self.ON_GROUND_CONFIRMATION_THRESHOLD = 0 
        self.UPWARD_VELOCITY_THRESHOLD = -25.0

        # --- General Parameters --//
        self.MAX_LOST_FRAMES = 15
        self.track_lost_count = 0
        
        # --- Gating Parameters ---
        self.MINIMUM_GATE_SIZE = 15.0

    def _calculate_average_velocity(self):
        if len(self.detection_history) < 2: return 50.0
        velocities = []
        history_list = list(self.detection_history)
        for i in range(len(history_list) - 1):
            frame1, pos1 = history_list[i]
            frame2, pos2 = history_list[i+1]
            delta_frames = frame2 - frame1
            if delta_frames > 0:
                distance = np.linalg.norm(pos2 - pos1)
                velocities.append(distance / delta_frames)
        return np.mean(velocities) if velocities else 50.0

    def update(self, detections, coord_transform, frame_count, dt, pitch_mask):
        """
        A single, unified update method. It either tracks a confirmed object
        or immediately acquires a new one.
        """
        if self.confirmed_track:
            self.confirmed_track.dt = dt
            predicted_pos = self.confirmed_track.predict()

            best_detection_sv = None
            if len(detections.xyxy) > 0:
                centers_abs = coord_transform.rel_to_abs(detections.get_anchors_coordinates(sv.Position.CENTER))
                
                safety_multiplier = 8.0 if self.ball_state == BallState.IN_AIR else 1.7
                avg_velocity = self._calculate_average_velocity()
                dynamic_gate = (avg_velocity * safety_multiplier) + self.MINIMUM_GATE_SIZE
                
                distances = np.linalg.norm(centers_abs - predicted_pos, axis=1)
                valid_indices = np.where(distances < dynamic_gate)[0]

                if len(valid_indices) > 0:
                    closest_idx = valid_indices[np.argmin(distances[valid_indices])]
                    best_detection_sv = detections[closest_idx:closest_idx+1]
            
            if best_detection_sv:
                self._update_ball_state(best_detection_sv, pitch_mask)
                center_to_update = coord_transform.rel_to_abs(best_detection_sv.get_anchors_coordinates(sv.Position.CENTER))
                self.confirmed_track.update(center_to_update[0])
                self.detection_history.append((frame_count, center_to_update[0]))
                self.track_lost_count = 0
                return best_detection_sv
            else:
                self.track_lost_count += 1
                if self.track_lost_count > self.MAX_LOST_FRAMES:
                    logger.warning(f"Track lost for {self.MAX_LOST_FRAMES} frames. Re-acquiring.")
                    self.confirmed_track = None
                    self.detection_history.clear()
                return None
        
        else: # self.confirmed_track is None, so we immediately re-acquire
            if len(detections.xyxy) > 0:
                logger.success("--- No active track. Performing Instant Lock ---")
                centers_abs = coord_transform.rel_to_abs(detections.get_anchors_coordinates(sv.Position.CENTER))
                best_idx = np.argmax(detections.confidence)

                self.confirmed_track = AdaptiveKalmanFilter(dt=dt)
                self.confirmed_track.initialize_state(centers_abs[best_idx])
                
                self.track_lost_count = 0
                self.detection_history.clear()
                self.detection_history.append((frame_count, centers_abs[best_idx]))
                
                return detections[best_idx:best_idx+1]
            return None # No track and no detections

    def _update_ball_state(self, detection_sv, pitch_mask):
        box = detection_sv.xyxy[0]
        center_x = int((box[0] + box[2]) / 2)
        bottom_y = int(box[3]) - 1

        h, w = pitch_mask.shape
        center_x = np.clip(center_x, 0, w - 1)
        bottom_y = np.clip(bottom_y, 0, h - 1)
        
        is_on_pitch = pitch_mask[bottom_y, center_x] == 255
        vertical_velocity = self.confirmed_track.x_hat[3, 0]

        if self.ball_state == BallState.ON_GROUND:
            if not is_on_pitch or vertical_velocity < self.UPWARD_VELOCITY_THRESHOLD:
                self.ball_state = BallState.IN_AIR
                self.on_ground_frames = 0
                logger.info(f"SWITCH TO IN_AIR (vy: {vertical_velocity:.2f}, on_pitch: {is_on_pitch})")
        
        elif self.ball_state == BallState.IN_AIR:
            if is_on_pitch and vertical_velocity >= 0:
                self.on_ground_frames += 1
            else:
                self.on_ground_frames = 0

            if self.on_ground_frames >= self.ON_GROUND_CONFIRMATION_THRESHOLD:
                self.ball_state = BallState.ON_GROUND
                logger.info("SWITCH TO ON_GROUND (Landed)")


class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.frame_queue = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        
        # --- MODIFIED FOR IMAGE DIRECTORY ---
        if not os.path.isdir(args.image_dir):
            raise IOError(f"Could not find image directory: {args.image_dir}")

        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
            raise IOError(f"No image files found in directory: {args.image_dir}")

        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape
        self.fps = args.fps
        logger.info(f"Image properties: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")
        # --- END MODIFICATION ---
        
        self.model = self.load_model()
        self.class_map, self.ball_class_id = self.load_class_map()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        self.ball_tracker = AdaptiveVelocityTracker()

    def load_model(self):
        try:
            logger.info(f"Loading model from checkpoint: {self.args.model_path}")
            model = RFDETRMedium(pretrain_weights=self.args.model_path)
            model.optimize_for_inference()
            logger.success("Model loaded and optimized successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_class_map(self):
        try:
            with open(self.args.annotation_path, "r") as f:
                coco_data = json.load(f)
            class_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
            ball_class_id = next((k for k, v in class_map.items() if v.lower() == 'ball'), None)
            if ball_class_id is None: raise ValueError("'ball' class not found in annotations.")
            logger.info(f"Loaded class map. Tracking 'ball' with ID: {ball_class_id}")
            return class_map, ball_class_id
        except Exception as e:
            logger.error(f"Error loading class map: {e}")
            raise

    def producer_thread(self):
        logger.info("Producer thread started.")
        # --- MODIFIED FOR IMAGE DIRECTORY ---
        for frame_count, image_path in enumerate(self.image_files):
            if self.stop_event.is_set():
                break
            
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Could not read image file: {image_path}")
                continue
            
            try:
                self.frame_queue.put((frame_count, frame), timeout=1)
            except queue.Full:
                time.sleep(0.1)
        
        self.frame_queue.put(None)
        logger.info("Producer thread finished.")
        # --- END MODIFICATION ---

    def consumer_thread(self):
        logger.info("Consumer thread started.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        # --- NEW: Open a file to save predictions ---
        with open(args.pred_name, "w") as predictions_file:
            while not self.stop_event.is_set():
                try:
                    item = self.frame_queue.get(timeout=1)
                    if item is None: break
                    
                    frame_count, frame = item

                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    pitch_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

                    coord_transform = self.motion_estimator.update(frame)
                    detections = self.model.predict(frame, confidence=self.args.confidence)
                    annotated_frame = frame.copy()
                    
                    ball_mask = detections.class_id == self.ball_class_id
                    ball_detections = detections[ball_mask]
                    
                    confirmed_detection_sv = self.ball_tracker.update(
                        ball_detections, coord_transform, frame_count, 1.0/self.fps, pitch_mask
                    )
                    
                    if confirmed_detection_sv:
                        # --- NEW: Write the confirmed detection to the file ---
                        box = confirmed_detection_sv.xyxy[0]
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        line = f"{frame_count+1},-1,{x1},{y1},{w},{h},1,-1,-1,-1\n"
                        predictions_file.write(line)

                        label = f"Ball ({self.ball_tracker.ball_state.name})"
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=confirmed_detection_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=confirmed_detection_sv, labels=[label])

                    out_writer.write(annotated_frame)
                    # cv2.imshow("Annotated Frame", annotated_frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.stop_event.set()
                    #     break
                
                except queue.Empty:
                    continue

        cv2.destroyAllWindows()
        out_writer.release()
        logger.info("Consumer thread finished.")

    def run(self):
        producer = threading.Thread(target=self.producer_thread)
        consumer = threading.Thread(target=self.consumer_thread)
        logger.info("Starting processing threads...")
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
        self.stop_event.set()
        logger.success("Processing complete.")

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

def main(args):
    try:
        processor = VideoProcessor(args)
        processor.run()
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid inference and tracking on a video.")
    # --- MODIFIED FOR IMAGE DIRECTORY ---
    parser.add_argument("--model_path", default="ball.pth", type=str)
    parser.add_argument("--image_dir",default="snmot196" , type=str, help="Path to the directory containing image frames.")
    parser.add_argument("--annotation_path", default="_annotations.coco.json", type=str)
    parser.add_argument("--output_path", type=str, default="output_hybrid4_snmot196.mp4")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the output video and Kalman filter.")
    parser.add_argument("--pred_name",type=str,default="predictions_hybrid4_snmot196.txt")

    # --- END MODIFICATION ---
    args = parser.parse_args()
    main(args)
