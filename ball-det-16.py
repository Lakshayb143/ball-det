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
# --- Paths ---
BALL_MODEL_PATH = "ball.pth"
PLAYER_MODEL_PATH = "player.pth" 
IMAGE_DIR_PATH = "img1"
OUTPUT_PATH = "output_v16.mp4"
PREDICTION_FILE_PATH = "predictions_v16.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.7
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 25
BALL_TRACKER_BUFFER_SIZE = 10 

# --- V16: Background Subtraction & Morphology Parameters ---
BS_HISTORY = 500 
BS_VAR_THRESHOLD = 16 
MIN_BALL_CONTOUR_AREA = 25
MAX_BALL_CONTOUR_AREA = 400
MIN_BALL_CIRCULARITY = 0.7 
BS_CANDIDATE_CONFIDENCE = 0.4 

# --- Optical Flow Parameters ---
MAX_OPTICAL_FLOW_GAP = 3

# --- Helper Functions ---
def is_point_in_boxes(point, boxes: np.ndarray) -> bool:
    if boxes.size == 0: return False
    return np.any((point[0] >= boxes[:, 0]) & (point[0] <= boxes[:, 2]) & (point[1] >= boxes[:, 1]) & (point[1] <= boxes[:, 3]))

class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)
    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0: return sv.Detections.empty()
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)
        if not self.buffer: return sv.Detections.empty()
        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]
    def reset(self):
        self.buffer.clear()

class AdaptiveKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        dt2 = 0.5 * dt ** 2
        self.A = np.array([[1, 0, dt, 0, dt2, 0], [0, 1, 0, dt, 0, dt2],[0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
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
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)])
        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape

        self.ball_model = RFDETRMedium(pretrain_weights=args.ball_model_path)
        self.player_model = RFDETRMedium(pretrain_weights=args.player_model_path)
        self.ball_model.optimize_for_inference()
        self.player_model.optimize_for_inference()
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        self.kf = AdaptiveKalmanFilter(dt=1.0 / args.fps)
        self.ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=BS_HISTORY, varThreshold=BS_VAR_THRESHOLD, detectShadows=False)
        
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
            for frame_count, image_path in enumerate(self.image_files, 1):
                frame = cv2.imread(image_path)
                if frame is None: continue

                annotated_frame = frame.copy()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                coord_transform = self.motion_estimator.update(frame)
                
                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                
                fg_mask = self.bg_subtractor.apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bs_boxes_xywh = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if MIN_BALL_CONTOUR_AREA < area < MAX_BALL_CONTOUR_AREA:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter == 0: continue
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if circularity > MIN_BALL_CIRCULARITY:
                            bs_boxes_xywh.append(cv2.boundingRect(cnt))
                
                # --- BUG FIX: Convert xywh boxes to xyxy format ---
                if bs_boxes_xywh:
                    xywh = np.array(bs_boxes_xywh)
                    xyxy = xywh.copy()
                    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] # x2 = x1 + w
                    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] # y2 = y1 + h
                    bs_detections = sv.Detections(xyxy=xyxy)
                else:
                    bs_detections = sv.Detections.empty()
                # --- END BUG FIX ---
                
                bs_detections.confidence = np.full(len(bs_detections), BS_CANDIDATE_CONFIDENCE)
                bs_detections.class_id = np.full(len(bs_detections), BALL_CLASS_ID)
                
                player_detections = self.player_model.predict(frame, confidence=self.args.player_confidence)
                player_detections = player_detections[np.isin(player_detections.class_id, PLAYER_CLASS_IDS)]
                
                rf_detections = self.ball_model.predict(frame, confidence=self.args.ball_confidence)
                rf_detections = rf_detections[rf_detections.class_id == BALL_CLASS_ID]
                
                all_ball_detections = sv.Detections.merge([rf_detections, bs_detections])
                
                valid_ball_indices = []
                if len(all_ball_detections) > 0:
                    ball_centers_rel = all_ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    for i, center in enumerate(ball_centers_rel):
                        if not is_point_in_boxes(center, player_detections.xyxy):
                            valid_ball_indices.append(i)
                
                filtered_ball_detections = all_ball_detections[valid_ball_indices]
                
                best_detection_sv = self.ball_tracker.update(filtered_ball_detections)
                
                measurement_abs, annotation_sv, annotation_label = None, None, ""
                if len(best_detection_sv) > 0:
                    center_rel = best_detection_sv.get_anchors_coordinates(sv.Position.CENTER)
                    measurement_abs = coord_transform.rel_to_abs(center_rel).flatten()
                    annotation_sv = best_detection_sv
                    annotation_label = "Ball"
                    self.optical_flow_gap_counter = 0

                if measurement_abs is None and self.track_initialized and self.optical_flow_gap_counter < MAX_OPTICAL_FLOW_GAP:
                    if self.optical_flow_points_rel is not None and self.prev_gray_frame is not None:
                        new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points_rel, None, **self.lk_params)
                        if status[0][0] == 1:
                            measurement_abs = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                            self.optical_flow_gap_counter += 1
                            annotation_label = f"Ball (OF)"
                            x_rel, y_rel = new_points_rel[0].ravel()
                            synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))

                if measurement_abs is not None:
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        self.kf.update(measurement_abs)
                    self.prev_position_abs = self.kf.x_hat[:2].flatten()
                    current_pos_rel = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
                    self.optical_flow_points_rel = np.array([[current_pos_rel]], dtype=np.float32)

                    if len(annotation_sv) > 0:
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[annotation_label])
                        box_to_write = annotation_sv.xyxy[0]
                        pred_file.write(f"{frame_count},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n")
                else:
                    self.track_initialized = False
                    self.optical_flow_points_rel = None
                    self.prev_position_abs = None

                out_writer.write(annotated_frame)
                self.prev_gray_frame = gray_frame.copy()

        out_writer.release()
        print("Processing complete for V16 (Corrected).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V16: Hybrid detection with Background Subtraction.")
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