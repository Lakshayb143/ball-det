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
# Using the parameters from your last provided script
BALL_MODEL_PATH = "ball.pth"
PLAYER_MODEL_PATH = "player.pth"
IMAGE_DIR_PATH = "img1"
OUTPUT_PATH = "output_ball-det.mp4"
PREDICTION_FILE_PATH = "temp.txt"
STATES_FILE_PATH = "states_ball-det.txt"
CONFIDENCE = 0.8
BALL_CONFIDENCE = 0.8
PLAYER_CONFIDENCE = 0.8
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 30
# How many pixels below the bbox to check for ground contact

GROUND_CLEARANCE_PIXELS = 9
# A less-strict velocity check, used in combination with other factors
UPWARD_VELOCITY_CONTEXT_THRESHOLD = -5.0 
ACTION_ZONE_HEIGHT_PERCENT = 0.2
ACTION_ZONE_WIDTH_EXPANSION_PERCENT = 0.25

# --- V17_stable: State Confirmation Thresholds ---

GROUND_CONFIRMATION_THRESHOLD = 10
AIR_CONFIRMATION_THRESHOLD = 10
OCCLUDED_CONFIRMATION_THRESHOLD = 4

# --- V17 Player Occlusion Mode Parameters ---

OCCLUSION_ENTRY_THRESHOLD = 25.0 
MAX_OCCLUSION_FRAMES = 8
MAX_LOST_FRAMES = 12
REACQUISITION_GATE_MULTIPLIER = 1.5

# --- Existing Tuning Parameters ---

Q_SCALE_DEFAULT = 1.0   
Q_SCALE_OCCLUDED = 13.0 
GRAVITY_PIXELS_PER_FRAME_SQUARED = 1.2

# Stricter threshold for high-velocity shots in open play

UPWARD_VELOCITY_SHOT_THRESHOLD = -35.0
GATE_SIZE_GROUND = 40.0
GATE_SIZE_AIR = 150.0

# Exclusion confidence penalty i.e. if the ball is detected in the same box as a player, reduce the confidence of the ball detection

EXCLUSION_CONFIDENCE_PENALTY = 0.0

# Ball Tracker Parameters
BALL_TRACKER_BUFFER_SIZE = 12


# ------------------------------------------------------------------------- 
VALIDATION_GATE_THRESHOLD = 25

# Outlier Detection Parameters
POSITION_THRESHOLD = 50.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 3

# Interpolation Parameters
INTERPOLATION_VELOCITY_THRESHOLD =  20.0  # Velocity threshold for interpolation
MAX_INTERPOLATION_FRAMES = 2  # Maximum consecutive frames to interpolate

# --- Optical Flow Parameters - ENHANCED ---
MAX_OPTICAL_FLOW_GAP = 3
OPTICAL_FLOW_ERROR_THRESHOLD =30.0  # Lower = stricter
OPTICAL_FLOW_MAX_MOVEMENT = 10.0    # Max pixels movement per frame
OPTICAL_FLOW_WIN_SIZE = (15, 15) # Increase for more stable tracking
OPTICAL_FLOW_MAX_LEVEL = 2           # Pyramid levels
OPTICAL_FLOW_CRITERIA_EPS = 0.03     # Lower = more precise
OPTICAL_FLOW_CRITERIA_COUNT = 10     # More iterations = more accurate


#-----------------------------------------------------------------------------------------------------------------------

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    OCCLUDED = 3 

# --- Helper Functions ---
def is_point_in_boxes(point, boxes: np.ndarray) -> bool:
    if boxes.size == 0: return False
    return np.any((point[0] >= boxes[:, 0]) & (point[0] <= boxes[:, 2]) & (point[1] >= boxes[:, 1]) & (point[1] <= boxes[:, 3]))

def find_nearby_player(point, player_detections, threshold):
    """Finds the closest player to a point, if within a threshold."""
    # --- BUG FIX: Add guard to prevent crash if point is None ---
    if point is None or len(player_detections) == 0:
        return None
        
    player_centers = player_detections.get_anchors_coordinates(sv.Position.CENTER)
    distances = np.linalg.norm(player_centers - point, axis=1)
    closest_player_idx = np.argmin(distances)
    if distances[closest_player_idx] < threshold:
        return player_detections[closest_player_idx:closest_player_idx+1]
    return None

def is_point_in_action_zones(point, player_boxes):
    for box in player_boxes:
        x1, y1, x2, y2 = box
        box_height, box_width = y2 - y1, x2 - x1
        action_y_start = y2 - (box_height * ACTION_ZONE_HEIGHT_PERCENT)
        width_expansion = box_width * ACTION_ZONE_WIDTH_EXPANSION_PERCENT
        action_x_start, action_x_end = x1 - width_expansion, x2 + width_expansion
        if (action_x_start <= point[0] <= action_x_end) and (action_y_start <= point[1] <= y2):
            return True
    return False

class PhysicsKalmanFilter:
    # ... (This class is unchanged) ...
    def __init__(self, dt=1.0, gravity_y=0.0):
        self.dt = dt; self.gravity_y = gravity_y
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 10.0
        self.x_hat = np.zeros((4, 1)); self.P = np.eye(4) * 100
        self.set_process_noise(Q_SCALE_DEFAULT) 
    def set_process_noise(self, q_scale): self.Q = np.eye(4) * q_scale
    def predict(self):
        self.x_hat = self.A @ self.x_hat; self.x_hat[3] += self.gravity_y * self.dt
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
        self.x_hat.fill(0.); self.x_hat[:2] = measurement.reshape(2, 1)
        if instant_velocity is not None: self.x_hat[2:] = instant_velocity.reshape(2, 1)
        self.P = np.eye(4) * 100

#---------------------------------------------------------------------------

class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return sv.Detections.empty()

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if not self.buffer:
            return sv.Detections.empty()

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

    def set_process_noise(self, accel_noise):
        self.Q[4, 4] = self.Q[5, 5] = accel_noise

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

class InterpolationTracker:
    """Handles interpolation of accepted predictions using a second Kalman filter."""
    
    def __init__(self, velocity_threshold=50.0, max_gap_frames=2):
        self.velocity_threshold = velocity_threshold
        self.max_gap_frames = max_gap_frames
        self.interpolation_kf = PhysicsKalmanFilter(dt=1.0/ FPS)
        self.accepted_positions = deque(maxlen=5)  # Store recent accepted positions
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0
        self.last_accepted_position = None
        self.last_accepted_velocity = None
        
    def add_accepted_prediction(self, position, velocity=None):
        """Add an accepted prediction to the interpolation tracker."""
        self.accepted_positions.append(position.copy())
        self.last_accepted_position = position.copy()
        if velocity is not None:
            self.last_accepted_velocity = velocity.copy()
        
        # Reset interpolation if we have a new accepted prediction
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0
        
    def should_interpolate(self, current_velocity=None):
        """Determine if interpolation should be active."""
        if len(self.accepted_positions) < 2:
            return False
            
        # Check velocity threshold if provided
        if current_velocity is not None:
            velocity_magnitude = np.linalg.norm(current_velocity)
            if velocity_magnitude > self.velocity_threshold:
                return False
                
        return True
        
    def start_interpolation(self):
        """Start interpolation process."""
        if len(self.accepted_positions) >= 2 and self.last_accepted_position is not None:
            self.interpolation_kf.initialize_state(self.last_accepted_position)
            self.interpolation_active = True
            self.interpolation_frames_remaining = self.max_gap_frames
            return True
        return False
        
    def get_interpolated_position(self):
        """Get the next interpolated position."""
        if not self.interpolation_active or self.interpolation_frames_remaining <= 0:
            return None
            
        predicted_pos = self.interpolation_kf.predict()
        self.interpolation_frames_remaining -= 1
        
        if self.interpolation_frames_remaining <= 0:
            self.interpolation_active = False
            
        return predicted_pos
        
    def stop_interpolation(self):
        """Stop the interpolation process."""
        self.interpolation_active = False
        self.interpolation_frames_remaining = 0

class OpticalKalmanFilter:
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

    def set_process_noise(self, accel_noise):
        self.Q[4, 4] = self.Q[5, 5] = accel_noise


#---------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)])
        first_frame = cv2.imread(self.image_files[0])
        self.frame_height, self.frame_width, _ = first_frame.shape
        self.ball_model = RFDETRMedium(pretrain_weights=args.ball_model_path)
        self.player_model = RFDETRMedium(pretrain_weights=args.player_model_path)
        self.ball_model.optimize_for_inference(); self.player_model.optimize_for_inference()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.kf = AdaptiveKalmanFilter(dt=1.0 / args.fps)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        self.track_initialized = False; self.lost_frames_count = 0
        self.ball_state = BallState.ON_GROUND
        self.ground_streak = 0; self.air_streak = 0; self.occluded_streak = 0

        #---------------------------------------------------------

        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        self.interpolation_tracker = InterpolationTracker(INTERPOLATION_VELOCITY_THRESHOLD, MAX_INTERPOLATION_FRAMES)


        self.ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)
        self.optical_flow_kf = OpticalKalmanFilter(dt=1.0 / args.fps)
        self.optical_kf_track_init = False


        self.track_initialized = False
        self.prev_position_abs = None
        self.track_lost_count = self.track_hit_streak = 0
        self.max_lost_frames = int(self.args.fps * 0.75)
        self.STABLE_TRACK_THRESHOLD = 5
        self.VALIDATION_GATE_THRESHOLD = 25
        
        self.optical_flow_points_rel = None
        self.prev_gray_frame = None
        self.optical_flow_gap_counter = 0
        self.lk_params = dict(
            winSize=OPTICAL_FLOW_WIN_SIZE, 
            maxLevel=OPTICAL_FLOW_MAX_LEVEL, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                      OPTICAL_FLOW_CRITERIA_COUNT, OPTICAL_FLOW_CRITERIA_EPS)
        )
    
    def _get_best_detection(self, ball_detections, predicted_pos, track_initialized, gate = VALIDATION_GATE_THRESHOLD):
        if len(ball_detections.xyxy) == 0:
            return None
        
        # Filter detections by confidence threshold
        high_conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(high_conf_mask):
            return None  # No detections meet confidence threshold
        
        filtered_detections = ball_detections[high_conf_mask]
        centers = filtered_detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if track_initialized and predicted_pos is not None:
            distances = np.linalg.norm(centers - predicted_pos, axis=1)
            valid_indices = np.where(distances < gate)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[np.argmin(distances[valid_indices])]
            else:
                # No detections within validation gate, select highest confidence
                idx = np.argmax(filtered_detections.confidence)
        else:
            idx = np.argmax(filtered_detections.confidence)
        
        return {"center": centers[idx], "sv": filtered_detections[idx:idx + 1]}

    def _process_detection(self, best_detection, track_initialized, frame_count):
        if not best_detection:
            return track_initialized, False, None
        
        current_position = best_detection["center"]
        current_velocity = current_position - self.prev_position_abs if self.prev_position_abs is not None else None
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
            
            # Add accepted prediction to interpolation tracker
            self.interpolation_tracker.add_accepted_prediction(current_position, current_velocity)
            
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position_abs = current_position.copy()
            print(f"Frame {frame_count}: ACCEPTED detection at {current_position}")
            
            return track_initialized, should_use_detection, best_detection
        else:
            self.track_lost_count += 1
            self.track_hit_streak = 0
            self.kf.set_process_noise(10.0)
            print(f"Frame {frame_count}: OUTLIER detected, rejecting prediction at {current_position}")
            
            if self.outlier_detector.should_reset_tracking():
                track_initialized = False
                self.track_lost_count = 0
                self.track_hit_streak = 0
                self.interpolation_tracker.stop_interpolation()
                print(f"Frame {frame_count}: Resetting tracking")
        
        return track_initialized, should_use_detection, None
    
    def _annotate_frame(self, annotated_frame, track_initialized, best_detection, should_use_detection, interpolated_position=None):
        # Annotate accepted detections (green/blue)
        if best_detection and should_use_detection and track_initialized and self.track_lost_count < self.max_lost_frames:
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=best_detection["sv"])
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=best_detection["sv"], labels=["Ball"])
        
        # Annotate interpolated predictions (red)
        if interpolated_position is not None:
            # Draw red circle for interpolated position
            cv2.circle(annotated_frame, (int(interpolated_position[0]), int(interpolated_position[1])), 10, (0, 0, 255), 2)  # Red circle
            cv2.putText(annotated_frame, "INTERPOLATED", (int(interpolated_position[0]) + 15, int(interpolated_position[1]) - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _manage_state_transitions(self, measurement_abs, annotation_sv, predicted_pos_abs, player_detections, pitch_mask, coord_transform):
        if not self.track_initialized: return

        has_measurement = measurement_abs is not None
        pos_to_check = measurement_abs if has_measurement else predicted_pos_abs
        is_lost_near_player = not has_measurement and find_nearby_player(pos_to_check, player_detections, OCCLUSION_ENTRY_THRESHOLD)
        
        if has_measurement and annotation_sv is not None:
            box = annotation_sv.xyxy[0]
            center_x = int((box[0] + box[2]) / 2)
            check_y = int(box[3] + GROUND_CLEARANCE_PIXELS)
            h, w = pitch_mask.shape
            center_x, check_y = np.clip(center_x, 0, w-1), np.clip(check_y, 0, h-1)
            is_on_ground = pitch_mask[check_y, center_x] == 255
            pos_to_check_rel = coord_transform.abs_to_rel(np.array([pos_to_check]))[0]
            vertical_velocity = self.kf.x_hat[3, 0]
            is_powerful_shot = vertical_velocity < UPWARD_VELOCITY_SHOT_THRESHOLD
            is_in_action_zone = is_point_in_action_zones(pos_to_check_rel, player_detections.xyxy)
            is_contextual_kick = is_in_action_zone and (vertical_velocity < UPWARD_VELOCITY_CONTEXT_THRESHOLD)

            if not is_on_ground or is_powerful_shot or is_contextual_kick:
                self.air_streak += 1; self.ground_streak, self.occluded_streak = 0, 0
            else:
                self.ground_streak += 1; self.air_streak, self.occluded_streak = 0, 0
            
        elif is_lost_near_player:
            self.occluded_streak += 1; self.ground_streak, self.air_streak = 0, 0
        
        if self.ground_streak >= GROUND_CONFIRMATION_THRESHOLD: self.ball_state = BallState.ON_GROUND
        elif self.air_streak >= AIR_CONFIRMATION_THRESHOLD: self.ball_state = BallState.IN_AIR
        elif self.occluded_streak >= OCCLUDED_CONFIRMATION_THRESHOLD: self.ball_state = BallState.OCCLUDED

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(self.args.output_path, fourcc, self.args.fps, (self.frame_width, self.frame_height))

        states_file = open(STATES_FILE_PATH, "w")
        
        with open(self.args.prediction_path, "w") as pred_file:
            for frame_count, image_path in enumerate(self.image_files, 1):
                frame = cv2.imread(image_path)
                if frame is None: continue
                annotated_frame = frame.copy()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


                valid_ball_indices = []
                
                if len(ball_detections) > 0 and len(player_detections) > 0:
                    ball_centers_rel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    penalized_confidences = ball_detections.confidence.copy()
                    for i, center in enumerate(ball_centers_rel):
                        if is_point_in_boxes(center, player_detections.xyxy):
                            penalized_confidences[i] *= EXCLUSION_CONFIDENCE_PENALTY
                        else:
                            valid_ball_indices.append(i)
                    ball_detections.confidence = penalized_confidences
                                            
                
                filtered_ball_detections = ball_detections[valid_ball_indices]

                
                if self.ball_state == BallState.OCCLUDED: gate = GATE_SIZE_AIR * REACQUISITION_GATE_MULTIPLIER
                elif self.ball_state == BallState.IN_AIR: gate = GATE_SIZE_AIR
                else: gate = GATE_SIZE_GROUND

                # Get best detection using the new outlier detection system
                best_detection = self._get_best_detection(filtered_ball_detections, predicted_pos_abs, self.track_initialized)
                self.track_initialized, should_use_detection, accepted_detection = self._process_detection(best_detection, self.track_initialized, frame_count)
                
                measurement_abs, annotation_sv, annotation_label = None, None, ""
                interpolated_position = None

# --------------------------------------------------------------------------------------------------------------
                # Having old tracker for optical flow (SEPARATE SYSTEM like in ball-det-96.py)

                detections_op = self.ball_tracker.update(filtered_ball_detections)
                measurement_abs_of = None

                if len(detections_op) > 0:
                    center_rel_of = detections_op.get_anchors_coordinates(sv.Position.CENTER)
                    measurement_abs_of = coord_transform.rel_to_abs(center_rel_of).flatten()
                    print(f"Frame {frame_count}: Optical flow detection at {center_rel_of}")

# --------------------------------------------------------------------------------------------------------------

                # Handle accepted detections from outlier detection
                if accepted_detection and should_use_detection:
                    center_rel = accepted_detection["center"]
                    measurement_abs = coord_transform.rel_to_abs(center_rel.reshape(1, -1)).flatten()
                    annotation_sv = accepted_detection["sv"]
                    annotation_label = "Ball (Detected)"
                    self.optical_flow_gap_counter = 0

                # Handle interpolation logic when no detection is accepted
                elif not best_detection and self.track_initialized:
                    # No detection found, check if we should interpolate
                    current_velocity = None
                    if self.prev_position_abs is not None and len(self.interpolation_tracker.accepted_positions) > 0:
                        last_accepted = self.interpolation_tracker.accepted_positions[-1]
                        current_velocity = self.prev_position_abs - last_accepted
                    
                    if self.interpolation_tracker.should_interpolate(current_velocity):
                        if not self.interpolation_tracker.interpolation_active:
                            self.interpolation_tracker.start_interpolation()
                            print(f"Frame {frame_count}: Starting interpolation")
                        
                        interpolated_position = self.interpolation_tracker.get_interpolated_position()
                        if interpolated_position is not None:
                            print(f"Frame {frame_count}: INTERPOLATED position at {interpolated_position}")
                            # Convert interpolated position to measurement
                            measurement_abs = coord_transform.rel_to_abs(interpolated_position.reshape(1, -1)).flatten()
                            # Create synthetic detection for annotation
                            x_rel, y_rel = interpolated_position
                            synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
                            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))
                            annotation_label = "Ball (Interpolated)"
                    else:
                        self.interpolation_tracker.stop_interpolation()
                    
                    self.track_lost_count += 1
                    self.track_hit_streak = 0
                    self.kf.set_process_noise(10.0)

                # Handle optical flow as fallback - EXACTLY like ball-det-96.py
                # if measurement_abs is None and measurement_abs_of is None and self.optical_kf_track_init and self.optical_flow_gap_counter < MAX_OPTICAL_FLOW_GAP:
                #     if self.optical_flow_points_rel is not None and self.prev_gray_frame is not None:
                #         new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.optical_flow_points_rel, None, **self.lk_params)
                #         if status[0][0] == 1:
                #             measurement_abs_of = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                #             self.optical_flow_gap_counter += 1
                #             annotation_label = f"Ball (OF) ({self.ball_state.name})"
                #             x_rel, y_rel = new_points_rel[0].ravel()
                #             synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
                #             annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))
                #             print(f"Frame {frame_count}: Optical flow tracking at {new_points_rel[0].ravel()}")
                #         else:
                #             print(f"Frame {frame_count}: Optical flow tracking lost")
                
                # Update optical flow Kalman filter
                if measurement_abs_of is not None:
                    if not self.optical_kf_track_init:
                        self.optical_flow_kf.initialize_state(measurement_abs_of)
                        self.optical_kf_track_init = True
                    else:
                        self.optical_flow_kf.update(measurement_abs_of)
                    self.prev_position_abs = self.optical_flow_kf.x_hat[:2].flatten()
                    current_pos_rel_of = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
                    self.optical_flow_points_rel = np.array([[current_pos_rel_of]], dtype=np.float32)
                else:
                    self.optical_kf_track_init = False
                    self.optical_flow_points_rel = None
                    self.prev_position_abs = None
                    if self.track_lost_count >= self.max_lost_frames and self.track_initialized:
                        self.track_hit_streak = 0
                        self.track_lost_count = 0

                # Handle occlusion case
                if self.ball_state == BallState.OCCLUDED and annotation_sv is None: 
                    occluding_player = find_nearby_player(predicted_pos_abs, player_detections, OCCLUSION_ENTRY_THRESHOLD * 1.5)
                    if occluding_player: measurement_abs = occluding_player.get_anchors_coordinates(sv.Position.CENTER)[0]

                # Update state transitions
                # if measurement_abs is not None:
                #     self._manage_state_transitions(measurement_abs, annotation_sv, predicted_pos_abs, player_detections, pitch_mask, coord_transform)
                # else:
                #     self._manage_state_transitions(measurement_abs_of, annotation_sv, predicted_pos_abs, player_detections, pitch_mask, coord_transform)

                self._manage_state_transitions(measurement_abs, annotation_sv, predicted_pos_abs, player_detections, pitch_mask, coord_transform)

                # Update main Kalman filter and handle tracking
                if measurement_abs is not None:
                    self.lost_frames_count = 0
                    noise_scale = Q_SCALE_OCCLUDED if self.ball_state == BallState.OCCLUDED else Q_SCALE_DEFAULT
                    self.kf.set_process_noise(noise_scale)
                    
                    if not self.track_initialized:
                        self.kf.initialize_state(measurement_abs)
                        self.track_initialized = True
                    else: 
                        self.kf.update(measurement_abs)
                    
                    if annotation_sv: 
                        pred_file.write(f"{frame_count},-1,{annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][1]},{annotation_sv.xyxy[0][2]-annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][3]-annotation_sv.xyxy[0][1]},1,-1,-1,-1\n")
                        state_num = self.ball_state.value
                        states_file.write(f"{frame_count},{state_num}\n")

                # elif measurement_abs_of is not None:
                #     # Write optical flow detections to file
                #     if annotation_sv is not None:
                #         box_to_write = annotation_sv.xyxy[0]
                #         line = f"{frame_count},-1,{box_to_write[0]},{box_to_write[1]},{box_to_write[2]-box_to_write[0]},{box_to_write[3]-box_to_write[1]},1,-1,-1,-1\n"
                #         pred_file.write(line)
                #         state_num = self.ball_state.value
                #         states_file.write(f"{frame_count},{state_num}\n")

                elif self.track_initialized:
                    self.lost_frames_count += 1
                    timeout = MAX_OCCLUSION_FRAMES if self.ball_state == BallState.OCCLUDED else MAX_LOST_FRAMES
                    if self.lost_frames_count > timeout:
                        self.track_initialized = False

                # Visualization
                if self.track_initialized or self.optical_kf_track_init:
                    final_pos = self.kf.x_hat[:2].flatten()
                    final_pos_rel = coord_transform.abs_to_rel(np.array([final_pos]))[0]
                    display_sv = annotation_sv
                    if display_sv is None:
                        x, y = final_pos_rel.ravel()
                        synthetic_box = np.array([x-10, y-10, x+10, y+10])
                        display_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))
                    
                    label = annotation_label + " (" + self.ball_state.name + ")"
                    # if measurement_abs_of is not None:
                    #     label = label + " (OF)"
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=display_sv, labels=[label])
                    self.box_annotator.color = sv.Color.YELLOW if self.ball_state == BallState.OCCLUDED else sv.Color.GREEN
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=display_sv)

                # Add real-time visualization with imshow
                display_frame = cv2.resize(annotated_frame, (960, 540))  # Resize for better display
                cv2.imshow('Ball Detection - Press Q to quit', display_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break

                out_writer.write(annotated_frame)
                self.prev_gray_frame = gray_frame.copy()
                print(f"Frame {frame_count} processed - State: {self.ball_state.name}")
        out_writer.release()
        states_file.close()
        cv2.destroyAllWindows()  # Clean up imshow windows
        print("Processing complete for ball-det.py with optical flow and real-time display.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V17_6: Stable tracker with state confirmation thresholds.")
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