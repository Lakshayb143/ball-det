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
# --- Paths ---
BALL_MODEL_PATH = "ball.pth"
PLAYER_MODEL_PATH = "player.pth"
IMAGE_DIR_PATH = "img1"
OUTPUT_PATH = "output_v17_stable.mp4"
PREDICTION_FILE_PATH = "predictions_v17_stable.txt"

# --- Model & Tracking Parameters ---
BALL_CONFIDENCE = 0.65 
PLAYER_CONFIDENCE = 0.5
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 30

# --- V17_stable: State Confirmation Thresholds ---
# How many consecutive frames of evidence are needed to confirm a state change
GROUND_CONFIRMATION_THRESHOLD = 4
AIR_CONFIRMATION_THRESHOLD = 7
OCCLUDED_CONFIRMATION_THRESHOLD = 5

# --- V17 Player Occlusion Mode Parameters ---
OCCLUSION_ENTRY_THRESHOLD = 30.0 
MAX_OCCLUSION_FRAMES = 5 # Timeout for the occlusion state
MAX_LOST_FRAMES = 6 # General timeout if not occluded
REACQUISITION_GATE_MULTIPLIER = 1.5 

# --- Existing Tuning Parameters ---
Q_SCALE_DEFAULT = 0.7   
Q_SCALE_OCCLUDED = 11.0 
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.4
EXCLUSION_CONFIDENCE_PENALTY = 0.0
UPWARD_VELOCITY_THRESHOLD = -20.0
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 150.0

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    OCCLUDED = 3 

# --- Helper Functions ---
def is_point_in_boxes(point, boxes: np.ndarray) -> bool:
    if boxes.size == 0: return False
    return np.any((point[0] >= boxes[:, 0]) & (point[0] <= boxes[:, 2]) & (point[1] >= boxes[:, 1]) & (point[1] <= boxes[:, 3]))

def find_nearby_player(point, player_detections, threshold):
    if len(player_detections) == 0: return None
    player_centers = player_detections.get_anchors_coordinates(sv.Position.CENTER)
    distances = np.linalg.norm(player_centers - point, axis=1)
    closest_player_idx = np.argmin(distances)
    if distances[closest_player_idx] < threshold:
        return player_detections[closest_player_idx:closest_player_idx+1]
    return None

class PhysicsKalmanFilter:
    # ... (This class is unchanged) ...
    def __init__(self, dt=1.0, gravity_y=0.0):
        self.dt = dt
        self.gravity_y = gravity_y
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 10.0
        self.x_hat = np.zeros((4, 1))
        self.P = np.eye(4) * 100
        self.set_process_noise(Q_SCALE_DEFAULT) 
    def set_process_noise(self, q_scale): self.Q = np.eye(4) * q_scale
    def predict(self):
        self.x_hat = self.A @ self.x_hat
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
    def initialize_state(self, measurement, instant_velocity=None):
        self.x_hat.fill(0.)
        self.x_hat[:2] = measurement.reshape(2, 1)
        if instant_velocity is not None:
            self.x_hat[2:] = instant_velocity.reshape(2, 1)
        self.P = np.eye(4) * 100

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

        self.kf = PhysicsKalmanFilter(dt=1.0 / args.fps, gravity_y=GRAVITY_PIXELS_PER_FRAME_SQUARED)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        
        self.track_initialized = False
        self.lost_frames_count = 0
        
        # --- V17_stable: State machine and streak counters ---
        self.ball_state = BallState.ON_GROUND
        self.ground_streak = 0
        self.air_streak = 0
        self.occluded_streak = 0

    def _manage_state_transitions(self, measurement_abs, predicted_pos_abs, player_detections, pitch_mask, coord_transform):
        """
        Gathers evidence and updates state counters. Confirms a state change only
        after a streak threshold is met (hysteresis).
        """
        if not self.track_initialized: return

        # --- 1. Gather Evidence for the current frame ---
        has_measurement = measurement_abs is not None
        is_lost_near_player = False
        is_on_pitch = False
        has_upward_velocity = False

        if has_measurement:
            pos_to_check = measurement_abs
        else: # If no measurement, use the prediction to check for occlusion
            pos_to_check = predicted_pos_abs
        
        pos_to_check_rel = coord_transform.abs_to_rel(np.array([pos_to_check]))[0]
        
        # Check if lost near a player (evidence for OCCLUDED)
        if not has_measurement:
            if find_nearby_player(pos_to_check, player_detections, OCCLUSION_ENTRY_THRESHOLD):
                is_lost_near_player = True
        
        # Check position and velocity (evidence for GROUND vs AIR)
        if has_measurement:
            x, y = int(pos_to_check_rel[0]), int(pos_to_check_rel[1])
            h, w = pitch_mask.shape
            x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
            is_on_pitch = pitch_mask[y, x] == 255
            vertical_velocity = self.kf.x_hat[3, 0]
            if vertical_velocity < UPWARD_VELOCITY_THRESHOLD:
                has_upward_velocity = True

        # --- 2. Update Streaks ("Voting") ---
        if is_lost_near_player:
            self.occluded_streak += 1
            self.ground_streak, self.air_streak = 0, 0
        elif has_measurement and (not is_on_pitch or has_upward_velocity):
            self.air_streak += 1
            self.ground_streak, self.occluded_streak = 0, 0
        elif has_measurement and is_on_pitch:
            self.ground_streak += 1
            self.air_streak, self.occluded_streak = 0, 0
        else: # True miss (not near player)
            self.ground_streak, self.air_streak, self.occluded_streak = 0, 0, 0

        # --- 3. Confirm State Change ("Decision") ---
        if self.occluded_streak >= OCCLUDED_CONFIRMATION_THRESHOLD:
            self.ball_state = BallState.OCCLUDED
        elif self.air_streak >= AIR_CONFIRMATION_THRESHOLD:
            self.ball_state = BallState.IN_AIR
        elif self.ground_streak >= GROUND_CONFIRMATION_THRESHOLD:
            self.ball_state = BallState.ON_GROUND
        # If no threshold is met, the state remains "sticky" to its last confirmed value.
        
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.args.fps, (self.frame_width, self.frame_height))
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files, 1):
                frame = cv2.imread(image_path)
                if frame is None: continue
                annotated_frame = frame.copy()
                coord_transform = self.motion_estimator.update(frame)
                
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
                color_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pitch_mask = np.zeros_like(color_mask)
                if contours:
                    pitch_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(pitch_contour) > (self.frame_width * self.frame_height * 0.1):
                        cv2.drawContours(pitch_mask, [pitch_contour], -1, 255, thickness=cv2.FILLED)

                predicted_pos_abs = self.kf.predict() if self.track_initialized else None
                player_detections = self.player_model.predict(frame, confidence=self.args.player_confidence)
                player_detections = player_detections[np.isin(player_detections.class_id, PLAYER_CLASS_IDS)]
                ball_detections = self.ball_model.predict(frame, confidence=self.args.ball_confidence)
                ball_detections = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
                
                measurement_abs, annotation_sv = None, None
                
                # Association logic remains largely the same, but uses the confirmed state
                if self.ball_state == BallState.OCCLUDED:
                    gate = GATE_SIZE_AIR * REACQUISITION_GATE_MULTIPLIER
                elif self.ball_state == BallState.IN_AIR:
                    gate = GATE_SIZE_AIR
                else: # ON_GROUND
                    gate = GATE_SIZE_GROUND

                if len(ball_detections) > 0 and self.track_initialized:
                    centers_abs = coord_transform.rel_to_abs(ball_detections.get_anchors_coordinates(sv.Position.CENTER))
                    distances = np.linalg.norm(centers_abs - predicted_pos_abs, axis=1)
                    valid_indices = np.where(distances < gate)[0]
                    if len(valid_indices) > 0:
                        best_idx = valid_indices[np.argmax(ball_detections.confidence[valid_indices])]
                        measurement_abs = centers_abs[best_idx]
                        annotation_sv = ball_detections[best_idx:best_idx + 1]
                elif len(ball_detections) > 0: # First frame
                    best_idx = np.argmax(ball_detections.confidence)
                    measurement_abs = coord_transform.rel_to_abs(ball_detections.get_anchors_coordinates(sv.Position.CENTER))[best_idx]
                    annotation_sv = ball_detections[best_idx:best_idx + 1]

                # If we are occluded, we can use the player as a pseudo-measurement
                if self.ball_state == BallState.OCCLUDED and measurement_abs is None:
                    occluding_player = find_nearby_player(predicted_pos_abs, player_detections, OCCLUSION_ENTRY_THRESHOLD * 1.5)
                    if occluding_player:
                        measurement_abs = occluding_player.get_anchors_coordinates(sv.Position.CENTER)[0]

                # Manage state transitions AFTER finding potential measurement
                self._manage_state_transitions(measurement_abs, predicted_pos_abs, player_detections, pitch_mask, coord_transform)

                if measurement_abs is not None:
                    self.lost_frames_count = 0
                    noise_scale = Q_SCALE_OCCLUDED if self.ball_state == BallState.OCCLUDED else Q_SCALE_DEFAULT
                    self.kf.set_process_noise(noise_scale)
                    
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else:
                        self.kf.update(measurement_abs)
                    
                    if annotation_sv: # Only write to file if it was a real ball detection
                        pred_file.write(f"{frame_count},-1,{annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][1]},{annotation_sv.xyxy[0][2]-annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][3]-annotation_sv.xyxy[0][1]},1,-1,-1,-1\n")
                        print(f"{frame_count}: Processed")
                
                elif self.track_initialized:
                    self.lost_frames_count += 1
                    timeout = MAX_OCCLUSION_FRAMES if self.ball_state == BallState.OCCLUDED else MAX_LOST_FRAMES
                    if self.lost_frames_count > timeout:
                        self.track_initialized = False
                
                if self.track_initialized:
                    final_pos = self.kf.x_hat[:2].flatten()
                    final_pos_rel = coord_transform.abs_to_rel(np.array([final_pos]))[0]
                    if annotation_sv is None:
                        x, y = final_pos_rel.ravel()
                        synthetic_box = np.array([x-10, y-10, x+10, y+10])
                        annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]),class_id=np.array([0]))
                    
                    label = f"Ball ({self.ball_state.name})"
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[label])
                    self.box_annotator.color = sv.Color.YELLOW if self.ball_state == BallState.OCCLUDED else sv.Color.GREEN
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)

                out_writer.write(annotated_frame)

        out_writer.release()
        print("Processing complete for V17 (Stable).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V17_stable: Ball tracking with state confirmation thresholds.")
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