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


# --- Configuration and Setup ---
logger.add("hybrid_tracker_final.log", rotation="10 MB", level="INFO")


class AdaptiveKalmanFilter:
    """
    A Kalman Filter with a Constant Acceleration model.
    The state is 6D: [x, y, vx, vy, ax, ay].
    """

    def __init__(self, dt=1.0):
        self.dt = dt
        dt2 = 0.5 * dt ** 2
        self.A = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.Q = np.eye(6)
        self.R = np.eye(2) * 5.0
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 100
        self.set_process_noise(accel_noise=10.0)  # Start with high responsiveness

    def set_process_noise(self, accel_noise):
        """Dynamically adjust process noise for acceleration."""
        self.Q[4, 4] = accel_noise
        self.Q[5, 5] = accel_noise

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
        logger.info(f"Kalman filter initialized at world coordinates: {measurement.flatten()}")


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

        # Read the first frame to get dimensions
        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape
        self.fps = args.fps
        # --- END MODIFICATION ---

        logger.info(f"Image properties: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")

        self.model = self.load_model()
        self.class_map, self.ball_class_id = self.load_class_map()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        self.kf = AdaptiveKalmanFilter(dt=1.0 / self.fps)
        self.track_lost_count = 0
        self.max_lost_frames = int(self.fps * 0.75)
        self.track_hit_streak = 0
        self.STABLE_TRACK_THRESHOLD = 5
        self.VALIDATION_GATE_THRESHOLD = 25

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
            if ball_class_id is None:
                raise ValueError("'ball' class not found in annotations.")
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
        
        self.frame_queue.put(None) # Signal that production is done
        logger.info("Producer thread finished.")
        # --- END MODIFICATION ---

    def consumer_thread(self):
        logger.info("Consumer thread started.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.fps,
                                     (self.frame_width, self.frame_height))

        track_initialized = False
        predictions_file = open(args.pred_name, "w")

        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    break

                frame_count, frame = item
                coord_transform = self.motion_estimator.update(frame)
                detections = self.model.predict(frame, confidence=self.args.confidence)
                annotated_frame = frame.copy()

                ball_mask = detections.class_id == self.ball_class_id
                ball_detections = detections[ball_mask]

                predicted_world_pos = self.kf.predict() if track_initialized else None
                best_detection = None

                if len(ball_detections.xyxy) > 0:
                    centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    centers_abs = coord_transform.rel_to_abs(centers_rel)

                    if track_initialized and predicted_world_pos is not None:
                        distances = np.linalg.norm(centers_abs - predicted_world_pos, axis=1)
                        valid_indices = np.where(distances < self.VALIDATION_GATE_THRESHOLD)[0]

                        if len(valid_indices) > 0:
                            closest_idx_in_valid = np.argmin(distances[valid_indices])
                            original_idx = valid_indices[closest_idx_in_valid]
                            best_detection = {
                                "center_abs": centers_abs[original_idx],
                                "sv": ball_detections[original_idx:original_idx + 1]
                            }
                    else:
                        best_idx = np.argmax(ball_detections.confidence)
                        best_detection = {
                            "center_abs": centers_abs[best_idx],
                            "sv": ball_detections[best_idx:best_idx + 1]
                        }

                if best_detection:
                    self.track_hit_streak += 1
                    self.track_lost_count = 0
                    if self.track_hit_streak > self.STABLE_TRACK_THRESHOLD:
                        self.kf.set_process_noise(1.0)

                    if not track_initialized:
                        self.kf.initialize_state(best_detection["center_abs"])
                        track_initialized = True
                    else:
                        self.kf.update(best_detection["center_abs"])

                elif track_initialized:
                    self.track_lost_count += 1
                    self.track_hit_streak = 0
                    self.kf.set_process_noise(10.0)

                if track_initialized and self.track_lost_count < self.max_lost_frames:
                    box_to_draw_sv = None
                    if best_detection:
                        box_to_draw_sv = best_detection["sv"]

                    if box_to_draw_sv:
                        box = box_to_draw_sv.xyxy[0]
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        line = f"{frame_count+1},-1,{x1},{y1},{w},{h},1,-1,-1,-1\n"
                        predictions_file.write(line)
                        
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=box_to_draw_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame,
                                                                        detections=box_to_draw_sv,
                                                                        labels=["Ball"])

                if self.track_lost_count >= self.max_lost_frames and track_initialized:
                    logger.warning(f"Track lost for too long at frame {frame_count}. Resetting tracker.")
                    track_initialized = False
                    self.track_hit_streak = 0
                    self.track_lost_count = 0

                out_writer.write(annotated_frame)

            except queue.Empty:
                continue

        predictions_file.close()
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


def main(args):
    try:
        processor = VideoProcessor(args)
        processor.run()
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and tracking on a directory of images.")
    # --- MODIFIED FOR IMAGE DIRECTORY ---
    parser.add_argument("--model_path", default="ball.pth", type=str)
    parser.add_argument("--image_dir", default="snmot200", type=str, help="Path to the directory containing image frames.")
    parser.add_argument("--annotation_path", default="_annotations.coco.json", type=str)
    parser.add_argument("--output_path", type=str, default="output_det6_snmot200.mp4")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the output video and Kalman filter.")
    parser.add_argument("--pred_name",type=str,default="predictions_det6_snmot200.txt")
    # --- END MODIFICATION ---
    args = parser.parse_args()
    main(args)
