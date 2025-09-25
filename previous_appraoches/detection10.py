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
logger.add("hybrid_tracker_final.log", rotation="10 MB", level="INFO")

class TrackerState(Enum):
    SEARCHING = 1
    CONFIRMED = 2

class ProbationaryHypothesis:
    """A class to hold a temporary, unconfirmed track with vetting properties."""
    def __init__(self, initial_detection, frame_count):
        self.kf = AdaptiveKalmanFilter(dt=1.0)
        self.kf.initialize_state(initial_detection["center_abs"])
        self.hit_streak = 1
        self.last_update_frame = frame_count
        self.last_sv = initial_detection["sv"]
        box = self.last_sv.xyxy[0]
        width, height = box[2] - box[0], box[3] - box[1]
        self.sizes = deque(maxlen=5)
        self.sizes.append(width * height)
        self.accelerations = deque(maxlen=5)
        self.accelerations.append(self.kf.x_hat[4:].flatten())

class AdaptiveVelocityTracker:
    """
    The main tracker class. It manages the overall state (SEARCHING, CONFIRMED)
    and uses an adaptive velocity gate for robust outlier rejection.
    """
    def __init__(self, confirmation_frames=3):
        self.state = TrackerState.SEARCHING
        self.confirmed_track = None
        self.probationary_hypotheses = []
        self.detection_history = deque(maxlen=5)

        self.CONFIRMATION_FRAMES = confirmation_frames
        self.MAX_LOST_FRAMES = 15
        self.track_lost_count = 0
        
        self.MIN_ASPECT_RATIO = 0.8
        self.MAX_ASPECT_RATIO = 1.2
        self.MAX_SIZE_CHANGE_RATIO = 1.5
        self.MAX_JERK = 15.0
        self.VELOCITY_SAFETY_MULTIPLIER = 1.5
        self.MINIMUM_GATE_SIZE = 70.0

    def _calculate_average_velocity(self):
        if len(self.detection_history) < 2:
            return 50.0
        
        velocities = []
        history_list = list(self.detection_history)
        for i in range(len(history_list) - 1):
            frame1, pos1 = history_list[i]
            frame2, pos2 = history_list[i+1]
            
            delta_frames = frame2 - frame1
            if delta_frames > 0:
                distance = np.linalg.norm(pos2 - pos1)
                velocities.append(distance / delta_frames)
        
        if not velocities:
            return 50.0
            
        return np.mean(velocities)

    def update(self, detections, coord_transform, frame_count, dt):
        if self.state == TrackerState.SEARCHING:
            return self._search_for_track(detections, coord_transform, frame_count, dt)
        elif self.state == TrackerState.CONFIRMED:
            return self._update_confirmed_track(detections, coord_transform, frame_count, dt)

    def _search_for_track(self, detections, coord_transform, frame_count, dt):
        if len(detections.xyxy) == 0:
            return None
        
        centers_rel = detections.get_anchors_coordinates(sv.Position.CENTER)
        centers_abs = coord_transform.rel_to_abs(centers_rel)

        unmatched_indices = list(range(len(detections.xyxy)))
        hypotheses_to_prune = []

        for i, hypo in enumerate(self.probationary_hypotheses):
            hypo.kf.dt = dt
            predicted_pos = hypo.kf.predict()
            
            distances = np.linalg.norm(centers_abs - predicted_pos, axis=1)
            valid_indices = np.where(distances < 100.0)[0]
            
            if len(valid_indices) > 0:
                best_match_idx_local = np.argmin(distances[valid_indices])
                best_match_idx = valid_indices[best_match_idx_local]
                if best_match_idx in unmatched_indices:
                    if self._vet_candidate(detections[best_match_idx:best_match_idx+1], hypo):
                        hypo.kf.update(centers_abs[best_match_idx])
                        hypo.hit_streak += 1
                        hypo.last_update_frame = frame_count
                        hypo.last_sv = detections[best_match_idx:best_match_idx+1]
                        unmatched_indices.remove(best_match_idx)
                    else:
                        hypotheses_to_prune.append(i)

        self.probationary_hypotheses = [
            h for i, h in enumerate(self.probationary_hypotheses) 
            if i not in hypotheses_to_prune and (frame_count - h.last_update_frame) < 5
        ]
        
        if unmatched_indices:
            unmatched_detections = detections[unmatched_indices]
            sorted_indices_local = np.argsort(unmatched_detections.confidence)[::-1]
            unmatched_indices_np = np.array(unmatched_indices)
            top_original_indices = unmatched_indices_np[sorted_indices_local[:2]]

            for idx in top_original_indices:
                box = detections.xyxy[idx]
                width, height = box[2] - box[0], box[3] - box[1]
                aspect_ratio = width / height if height > 0 else 0
                if self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO:
                    new_detection = {"center_abs": centers_abs[idx], "sv": detections[idx:idx+1]}
                    self.probationary_hypotheses.append(ProbationaryHypothesis(new_detection, frame_count))
        
        for i, hypo in enumerate(self.probationary_hypotheses):
            if hypo.hit_streak >= self.CONFIRMATION_FRAMES:
                logger.success(f"--- Track Confirmed (Streak: {hypo.hit_streak}) ---")
                self.state = TrackerState.CONFIRMED
                self.confirmed_track = hypo.kf
                self.track_lost_count = 0
                self.detection_history.clear()
                self.detection_history.append((frame_count, hypo.kf.x_hat[:2].flatten()))
                return hypo.last_sv

        return None

    def _update_confirmed_track(self, detections, coord_transform, frame_count, dt):
        self.confirmed_track.dt = dt
        predicted_pos = self.confirmed_track.predict()
        
        avg_velocity = self._calculate_average_velocity()
        dynamic_gate = (avg_velocity * self.VELOCITY_SAFETY_MULTIPLIER) + self.MINIMUM_GATE_SIZE

        if len(detections.xyxy) > 0:
            centers_rel = detections.get_anchors_coordinates(sv.Position.CENTER)
            centers_abs = coord_transform.rel_to_abs(centers_rel)
            
            last_confirmed_pos = self.detection_history[-1][1]
            distances = np.linalg.norm(centers_abs - last_confirmed_pos, axis=1)
            valid_indices = np.where(distances < dynamic_gate)[0]

            if len(valid_indices) > 0:
                kf_distances = np.linalg.norm(centers_abs[valid_indices] - predicted_pos, axis=1)
                best_local_idx = np.argmin(kf_distances)
                original_idx = valid_indices[best_local_idx]

                self.confirmed_track.update(centers_abs[original_idx])
                self.detection_history.append((frame_count, centers_abs[original_idx]))
                self.track_lost_count = 0
                return detections[original_idx:original_idx+1]

        self.track_lost_count += 1
        if self.track_lost_count > self.MAX_LOST_FRAMES:
            logger.warning(f"Confirmed track lost at frame {frame_count}. Resetting to SEARCHING.")
            self.state = TrackerState.SEARCHING
            self.confirmed_track = None
            self.probationary_hypotheses = []
            self.detection_history.clear()
        
        return None

    def _vet_candidate(self, detection_sv, hypothesis):
        box = detection_sv.xyxy[0]
        width, height = box[2] - box[0], box[3] - box[1]
        
        aspect_ratio = width / height if height > 0 else 0
        if not (self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO): return False

        avg_size = np.mean(hypothesis.sizes)
        new_size = width * height
        size_change_ratio = new_size / avg_size if avg_size > 0 else 1.0
        if size_change_ratio > self.MAX_SIZE_CHANGE_RATIO or size_change_ratio < (1 / self.MAX_SIZE_CHANGE_RATIO): return False

        current_accel = hypothesis.kf.x_hat[4:].flatten()
        if len(hypothesis.accelerations) > 0:
            jerk = np.linalg.norm(current_accel - hypothesis.accelerations[-1])
            if jerk > self.MAX_JERK: return False
        
        hypothesis.sizes.append(new_size)
        hypothesis.accelerations.append(current_accel)
        return True


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
        
        # --- NEW: PREDICTION FILE LOGIC ---
        with open(args.pred_name, "w") as predictions_file:
            while not self.stop_event.is_set():
                try:
                    item = self.frame_queue.get(timeout=1)
                    if item is None: break
                    
                    frame_count, frame = item
                    
                    coord_transform = self.motion_estimator.update(frame)
                    detections = self.model.predict(frame, confidence=self.args.confidence)
                    annotated_frame = frame.copy()
                    
                    ball_mask = detections.class_id == self.ball_class_id
                    ball_detections = detections[ball_mask]
                    
                    confirmed_detection_sv = self.ball_tracker.update(ball_detections, coord_transform, frame_count, 1.0/self.fps)
                    
                    if confirmed_detection_sv:
                        # --- Write prediction to file ---
                        box = confirmed_detection_sv.xyxy[0]
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        line = f"{frame_count+1},-1,{x1},{y1},{w},{h},1,-1,-1,-1\n"
                        predictions_file.write(line)

                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=confirmed_detection_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=confirmed_detection_sv, labels=["Ball"])

                    out_writer.write(annotated_frame)
                    
                except queue.Empty:
                    continue

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
    parser = argparse.ArgumentParser(description="Run robust inference and tracking on a directory of images.")
    # --- MODIFIED FOR IMAGE DIRECTORY ---
    parser.add_argument("--model_path", default="ball.pth", type=str)
    parser.add_argument("--annotation_path", default="_annotations.coco.json", type=str)
    parser.add_argument("--image_dir", default="snmot200", type=str, help="Path to the directory containing image frames.")
    parser.add_argument("--pred_name",type=str,default="predictions_det10_snmot200.txt")
    parser.add_argument("--output_path", type=str, default="output_det10_snmot200.mp4")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the output video and Kalman filter.")

    # --- END MODIFICATION ---
    args = parser.parse_args()
    main(args)