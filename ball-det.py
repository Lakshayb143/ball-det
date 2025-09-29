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
OUTPUT_PATH = "output_ball-det-2.mp4"
PREDICTION_FILE_PATH = "temp.txt"
STATES_FILE_PATH = "states_ball-det-2.txt"
CONFIDENCE = 0.8
BALL_CONFIDENCE = 0.8
PLAYER_CONFIDENCE = 0.8
BALL_CLASS_ID = 0
PLAYER_CLASS_IDS = [1, 2, 3] 
FPS = 30
# How many pixels below the bbox to check for ground contact

GROUND_CLEARANCE_PIXELS = 12
# A less-strict velocity check, used in combination with other factors
UPWARD_VELOCITY_CONTEXT_THRESHOLD = -5.0 
ACTION_ZONE_HEIGHT_PERCENT = 0.2
ACTION_ZONE_WIDTH_EXPANSION_PERCENT = 0.25


# --- Smart State Transition Thresholds ---
# Different thresholds based on current state -> target state
GROUND_TO_AIR_THRESHOLD = 3      # Fast transition when ball leaves ground
GROUND_TO_OCCLUDED_THRESHOLD = 5  # Slower, need more evidence
AIR_TO_GROUND_THRESHOLD = 10     # Normal threshold
AIR_TO_OCCLUDED_THRESHOLD = 20   # Very high - avoid air->occluded transitions
OCCLUDED_TO_ANY_THRESHOLD = 2    # Normal recovery from occlusion

# --- V17 Player Occlusion Mode Parameters ---

OCCLUSION_ENTRY_THRESHOLD = 50.0  # Increased from 25.0
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
BALL_TRACKER_BUFFER_SIZE = 14


# ------------------------------------------------------------------------- 
VALIDATION_GATE_THRESHOLD = 25

# Outlier Detection Parameters
POSITION_THRESHOLD = 40.0
VELOCITY_THRESHOLD = 100.0
HISTORY_FRAMES = 4

# Outlier Detection for Interpolation/Optical Flow
INTERPOLATION_VALIDATION_GATE = 15.0  # Validation gate for interpolated positions
OPTICAL_FLOW_VALIDATION_GATE = 25.0   # Validation gate for optical flow positions

# Interpolation Parameters
INTERPOLATION_VELOCITY_THRESHOLD =  20.0  # Velocity threshold for interpolation
MAX_INTERPOLATION_FRAMES = 2  # Maximum consecutive frames to interpolate

# --- Optical Flow Parameters - ENHANCED ---
MAX_OPTICAL_FLOW_GAP = 9
OPTICAL_FLOW_ERROR_THRESHOLD =20.0  # Lower = stricter
OPTICAL_FLOW_MAX_MOVEMENT = 12.0    # Max pixels movement per frame
OPTICAL_FLOW_WIN_SIZE = (16, 16) # Increase for more stable tracking
OPTICAL_FLOW_MAX_LEVEL = 2           # Pyramid levels
OPTICAL_FLOW_CRITERIA_EPS = 0.02     # Lower = more precise
OPTICAL_FLOW_CRITERIA_COUNT = 12     # More iterations = more accurate


#-----------------------------------------------------------------------------------------------------------------------

class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    OCCLUDED = 3

class TrackingMode(Enum):
    DETECTION = 1      # Normal detection working
    INTERPOLATION = 2  # Using interpolation fallback  
    OPTICAL_FLOW = 3   # Using optical flow fallback
    LOST = 4          # All systems failed, trying to reacquire 

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
        self.interpolation_kf = EnhancedPhysicsKalmanFilter(dt=1.0/ FPS)
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


class EnhancedPhysicsKalmanFilter:
    def __init__(self, dt=1.0, gravity_y=GRAVITY_PIXELS_PER_FRAME_SQUARED):
        self.dt = dt
        self.gravity_y = gravity_y
        
        # 6-state vector: [x, y, vx, vy, ax, ay]
        dt2 = 0.5 * dt * dt
        self.A = np.array([
            [1, 0, dt, 0,  dt2, 0  ],  # x position
            [0, 1, 0,  dt, 0,   dt2],  # y position  
            [0, 0, 1,  0,  dt,  0  ],  # x velocity
            [0, 0, 0,  1,  0,   dt ],  # y velocity
            [0, 0, 0,  0,  1,   0  ],  # x acceleration
            [0, 0, 0,  0,  0,   1  ]   # y acceleration
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Observe x position
            [0, 1, 0, 0, 0, 0]   # Observe y position
        ])
        
        # State vector: [x, y, vx, vy, ax, ay]
        self.x_hat = np.zeros((6, 1))
        self.P = np.eye(6) * 100
        
        # Measurement noise (position uncertainty)
        self.R = np.eye(2) * 10.0
        
        # Process noise - tunable per state
        self.set_process_noise(1.0)
    
    def set_process_noise(self, base_scale):
        """Adaptive noise based on ball state"""
        pos_noise = base_scale * 0.5      # Position noise
        vel_noise = base_scale * 2.0      # Velocity noise  
        acc_noise = base_scale * 5.0      # Acceleration noise
        
        self.Q = np.diag([pos_noise, pos_noise, vel_noise, vel_noise, acc_noise, acc_noise])
    
    def predict(self):
        """Predict with gravity and acceleration"""
        # Apply state transition
        self.x_hat = self.A @ self.x_hat
        
        # Add gravity to y-velocity
        self.x_hat[3] += self.gravity_y * self.dt
        
        # Update covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x_hat[:2].flatten()
    
    def update(self, measurement):
        """Update with measurement"""
        measurement = measurement.reshape(2, 1)
        
        # Innovation
        y = measurement - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def initialize_state(self, measurement, velocity=None, acceleration=None):
        """Initialize with position, optional velocity and acceleration"""
        self.x_hat.fill(0.)
        self.x_hat[:2] = measurement.reshape(2, 1)
        
        if velocity is not None:
            self.x_hat[2:4] = velocity.reshape(2, 1)
        
        if acceleration is not None:
            self.x_hat[4:6] = acceleration.reshape(2, 1)
            
        self.P = np.eye(6) * 100
    
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.x_hat[2:4].flatten()
    
    def get_acceleration(self):
        """Get current acceleration estimate"""
        return self.x_hat[4:6].flatten()
#---------------------------------------------------------------------------

class UnifiedTracker:
    """Hierarchical tracker: Detection → Interpolation → Optical Flow → Lost → Reset"""
    
    def __init__(self, fps=30):
        self.mode = TrackingMode.DETECTION
        self.interpolation_frames_used = 0
        self.optical_flow_frames_used = 0
        self.total_lost_frames = 0
        
        # Initialize all tracking components
        self.ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)
        self.interpolation_tracker = InterpolationTracker(INTERPOLATION_VELOCITY_THRESHOLD, MAX_INTERPOLATION_FRAMES)
        self.optical_flow_kf = EnhancedPhysicsKalmanFilter(dt=1.0/fps)
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        
        # Optical flow state
        self.optical_flow_points_rel = None
        self.prev_gray_frame = None
        self.optical_kf_track_init = False
        self.prev_position_abs = None
        
        # Tracking state
        self.track_hit_streak = 0
        self.track_lost_count = 0
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=OPTICAL_FLOW_WIN_SIZE, 
            maxLevel=OPTICAL_FLOW_MAX_LEVEL, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                     OPTICAL_FLOW_CRITERIA_COUNT, OPTICAL_FLOW_CRITERIA_EPS)
        )
    
    def get_measurement(self, filtered_detections, predicted_pos_abs, track_initialized, 
                       coord_transform, gray_frame, frame_count):
        """Main method that returns measurement based on hierarchical fallback"""
        
        # Always update ball_tracker for optical flow readiness
        detections_op = self.ball_tracker.update(filtered_detections)
        
        measurement_abs = None
        annotation_sv = None
        annotation_label = ""
        
        if self.mode == TrackingMode.DETECTION:
            measurement_abs, annotation_sv, annotation_label = self._try_detection(
                filtered_detections, predicted_pos_abs, track_initialized, frame_count)
            
            if measurement_abs is None:
                self._switch_to_optical_flow(detections_op, coord_transform, frame_count)
                
                
        elif self.mode == TrackingMode.INTERPOLATION:
            measurement_abs, annotation_sv, annotation_label = self._try_interpolation(
                coord_transform, frame_count)
            
            if measurement_abs is None:  # Interpolation exhausted
                self._switch_to_lost(frame_count)
                
        elif self.mode == TrackingMode.OPTICAL_FLOW:
            measurement_abs, annotation_sv, annotation_label = self._try_optical_flow(
                detections_op, coord_transform, gray_frame, frame_count)
            
            if measurement_abs is None:  # Optical flow exhausted
                self._switch_to_interpolation(frame_count)
                
        elif self.mode == TrackingMode.LOST:
            # Try to reacquire with detection
            measurement_abs, annotation_sv, annotation_label = self._try_detection(
                filtered_detections, predicted_pos_abs, track_initialized, frame_count)
            
            if measurement_abs is not None:
                self._reset_and_switch_to_detection(frame_count)
            else:
                self.total_lost_frames += 1
                if self.total_lost_frames > MAX_LOST_FRAMES:
                    self._full_reset(frame_count)
        
        # Store gray frame for next optical flow
        self.prev_gray_frame = gray_frame.copy() if gray_frame is not None else None
        
        return measurement_abs, annotation_sv, annotation_label
    
    def _try_detection(self, filtered_detections, predicted_pos_abs, track_initialized, frame_count):
        """Try normal detection with outlier detection"""
        best_detection = self._get_best_detection(filtered_detections, predicted_pos_abs, track_initialized)
        
        if not best_detection:
            return None, None, ""
        
        current_position = best_detection["center"]
        current_velocity = current_position - self.prev_position_abs if self.prev_position_abs is not None else None
        is_outlier, should_use_detection = self.outlier_detector.is_outlier(current_position, current_velocity)
        
        if should_use_detection:
            self.track_hit_streak += 1
            self.track_lost_count = 0
            
            # Add accepted prediction to interpolation tracker for future use
            self.interpolation_tracker.add_accepted_prediction(current_position, current_velocity)
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position_abs = current_position.copy()
            
            print(f"Frame {frame_count}: DETECTION accepted at {current_position}")
            return current_position, best_detection["sv"], "Ball (Detected)"
        else:
            self.track_hit_streak = 0
            self.track_lost_count += 1
            print(f"Frame {frame_count}: DETECTION rejected (outlier) at {current_position}")
            
            if self.outlier_detector.should_reset_tracking():
                self._full_reset(frame_count)
                print(f"Frame {frame_count}: Outlier detector triggered full reset")
            
            return None, None, ""
    
    def _try_interpolation(self, coord_transform, frame_count):
        """Try interpolation fallback with outlier detection"""
        if self.interpolation_frames_used >= MAX_INTERPOLATION_FRAMES:
            return None, None, ""
        
        if not self.interpolation_tracker.interpolation_active:
            if not self.interpolation_tracker.start_interpolation():
                return None, None, ""
        
        interpolated_position = self.interpolation_tracker.get_interpolated_position()
        if interpolated_position is not None:
            measurement_abs = coord_transform.rel_to_abs(interpolated_position.reshape(1, -1)).flatten()
            
            # Validate interpolated position against last known good position
            if self.prev_position_abs is not None:
                distance = np.linalg.norm(measurement_abs - self.prev_position_abs)
                if distance > INTERPOLATION_VALIDATION_GATE:
                    print(f"Frame {frame_count}: INTERPOLATION rejected - too far from last position ({distance:.1f} > {INTERPOLATION_VALIDATION_GATE})")
                    return None, None, ""
            
            self.interpolation_frames_used += 1
            
            # Create synthetic annotation
            x_rel, y_rel = interpolated_position
            synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))
            
            print(f"Frame {frame_count}: INTERPOLATION accepted {self.interpolation_frames_used}/{MAX_INTERPOLATION_FRAMES}")
            return measurement_abs, annotation_sv, "Ball (Interpolated)"
        
        return None, None, ""
    
    def _try_optical_flow(self, detections_op, coord_transform, gray_frame, frame_count):
        """Try optical flow fallback with outlier detection"""
        if self.optical_flow_frames_used >= MAX_OPTICAL_FLOW_GAP:
            return None, None, ""
        
        measurement_abs_of = None
        
        # Try detection-based optical flow first
        if len(detections_op) > 0:
            center_rel_of = detections_op.get_anchors_coordinates(sv.Position.CENTER)
            measurement_abs_of = coord_transform.rel_to_abs(center_rel_of).flatten()
            print(f"Frame {frame_count}: Optical flow detection at {center_rel_of}")
        
        # Try Lucas-Kanade optical flow if no detection
        elif (self.optical_flow_points_rel is not None and 
              self.prev_gray_frame is not None and gray_frame is not None):
            
            new_points_rel, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_frame, gray_frame, 
                self.optical_flow_points_rel, None, **self.lk_params)
            
            if status[0][0] == 1:
                measurement_abs_of = coord_transform.rel_to_abs(new_points_rel[0]).flatten()
                print(f"Frame {frame_count}: Optical flow LK tracking")
        
        if measurement_abs_of is not None:
            # Validate optical flow position against last known good position
            if self.prev_position_abs is not None:
                distance = np.linalg.norm(measurement_abs_of - self.prev_position_abs)
                if distance > OPTICAL_FLOW_VALIDATION_GATE:
                    print(f"Frame {frame_count}: OPTICAL_FLOW rejected - too far from last position ({distance:.1f} > {OPTICAL_FLOW_VALIDATION_GATE})")
                    return None, None, ""
            
            self.optical_flow_frames_used += 1
            
            # Update optical flow Kalman filter
            if not self.optical_kf_track_init:
                self.optical_flow_kf.initialize_state(measurement_abs_of)
                self.optical_kf_track_init = True
            else:
                self.optical_flow_kf.update(measurement_abs_of)
            
            # Update tracking point for next frame
            self.prev_position_abs = self.optical_flow_kf.x_hat[:2].flatten()
            current_pos_rel_of = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
            self.optical_flow_points_rel = np.array([[current_pos_rel_of]], dtype=np.float32)
            
            # Create synthetic annotation
            x_rel, y_rel = current_pos_rel_of
            synthetic_box = np.array([x_rel-10, y_rel-10, x_rel+10, y_rel+10])
            annotation_sv = sv.Detections(xyxy=np.array([synthetic_box]), class_id=np.array([0]))
            
            print(f"Frame {frame_count}: OPTICAL_FLOW accepted {self.optical_flow_frames_used}/{MAX_OPTICAL_FLOW_GAP}")
            return measurement_abs_of, annotation_sv, "Ball (Optical Flow)"
        
        return None, None, ""
    
    def _switch_to_interpolation(self, frame_count):
        """Switch from detection to interpolation mode"""
        self.mode = TrackingMode.INTERPOLATION
        self.interpolation_frames_used = 0
        print(f"Frame {frame_count}: Switching to INTERPOLATION mode")
    
    def _switch_to_optical_flow(self, detections_op, coord_transform, frame_count):
        """Switch from interpolation to optical flow mode"""
        self.mode = TrackingMode.OPTICAL_FLOW
        self.optical_flow_frames_used = 0
        
        # Initialize optical flow if we have detection
        if len(detections_op) > 0 and self.prev_position_abs is not None:
            current_pos_rel_of = coord_transform.abs_to_rel(np.array([self.prev_position_abs]))[0]
            self.optical_flow_points_rel = np.array([[current_pos_rel_of]], dtype=np.float32)
        
        print(f"Frame {frame_count}: Switching to OPTICAL_FLOW mode")
    
    def _switch_to_lost(self, frame_count):
        """Switch from optical flow to lost mode"""
        self.mode = TrackingMode.LOST
        self.total_lost_frames = 0
        print(f"Frame {frame_count}: Switching to LOST mode")
    
    def _reset_and_switch_to_detection(self, frame_count):
        """Reset everything and switch back to detection mode"""
        self.mode = TrackingMode.DETECTION
        self.interpolation_frames_used = 0
        self.optical_flow_frames_used = 0
        self.total_lost_frames = 0
        self.track_hit_streak = 0
        self.track_lost_count = 0
        print(f"Frame {frame_count}: REACQUIRED - Switching back to DETECTION mode")
    
    def _full_reset(self, frame_count):
        """Full reset when max lost frames exceeded"""
        self._reset_and_switch_to_detection(frame_count)
        self.optical_kf_track_init = False
        self.optical_flow_points_rel = None
        self.prev_position_abs = None
        self.interpolation_tracker.stop_interpolation()
        self.ball_tracker.reset()
        self.outlier_detector = OutlierDetector(POSITION_THRESHOLD, VELOCITY_THRESHOLD, HISTORY_FRAMES)
        print(f"Frame {frame_count}: FULL RESET - All trackers reset")
    
    def _get_best_detection(self, ball_detections, predicted_pos, track_initialized):
        """Get best detection (reuse existing logic)"""
        if len(ball_detections.xyxy) == 0:
            return None
        
        high_conf_mask = ball_detections.confidence >= CONFIDENCE
        if not np.any(high_conf_mask):
            return None
        
        filtered_detections = ball_detections[high_conf_mask]
        centers = filtered_detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if track_initialized and predicted_pos is not None:
            distances = np.linalg.norm(centers - predicted_pos, axis=1)
            valid_indices = np.where(distances < VALIDATION_GATE_THRESHOLD)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[np.argmin(distances[valid_indices])]
            else:
                idx = np.argmax(filtered_detections.confidence)
        else:
            idx = np.argmax(filtered_detections.confidence)
        
        return {"center": centers[idx], "sv": filtered_detections[idx:idx + 1]}

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
        self.kf = EnhancedPhysicsKalmanFilter(dt=1.0 / args.fps)
        self.motion_estimator = MotionEstimator(transformations_getter=HomographyTransformationGetter())
        self.track_initialized = False; self.lost_frames_count = 0
        self.ball_state = BallState.ON_GROUND
        self.ground_streak = 0; self.air_streak = 0; self.occluded_streak = 0

        #---------------------------------------------------------
        # Initialize the unified hierarchical tracker
        self.unified_tracker = UnifiedTracker(fps=args.fps)
        
        # Keep track initialization for main Kalman filter
        self.track_initialized = False


    def update_filter_for_state(self, ball_state):
        if ball_state == BallState.ON_GROUND:
            self.kf.set_process_noise(1.0)  # Low noise - predictable motion
        elif ball_state == BallState.IN_AIR:
            self.kf.set_process_noise(4.0)  # Higher noise - less predictable
        elif ball_state == BallState.OCCLUDED:
            self.kf.set_process_noise(8.0)  # Highest noise - very uncertain
    

    

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
            # is_powerful_shot = vertical_velocity < UPWARD_VELOCITY_SHOT_THRESHOLD
            # is_in_action_zone = is_point_in_action_zones(pos_to_check_rel, player_detections.xyxy)
            # is_contextual_kick = is_in_action_zone and (vertical_velocity < UPWARD_VELOCITY_CONTEXT_THRESHOLD)

            if not is_on_ground:
                self.air_streak += 1; self.ground_streak, self.occluded_streak = 0, 0
            else:
                self.ground_streak += 1; self.air_streak, self.occluded_streak = 0, 0
            
        elif is_lost_near_player:
            self.occluded_streak += 1; self.ground_streak, self.air_streak = 0, 0
        
        # Smart state transitions based on current state
        current_state = self.ball_state
        
        if current_state == BallState.ON_GROUND:
            # From GROUND: Fast transition to AIR, slower to OCCLUDED
            if self.air_streak >= GROUND_TO_AIR_THRESHOLD:
                self.ball_state = BallState.IN_AIR
            elif self.occluded_streak >= GROUND_TO_OCCLUDED_THRESHOLD:
                self.ball_state = BallState.OCCLUDED
                
        elif current_state == BallState.IN_AIR:
            # From AIR: Normal transition to GROUND, very high threshold for OCCLUDED
            if self.ground_streak >= AIR_TO_GROUND_THRESHOLD:
                self.ball_state = BallState.ON_GROUND
            elif self.occluded_streak >= AIR_TO_OCCLUDED_THRESHOLD:
                self.ball_state = BallState.OCCLUDED
                
        elif current_state == BallState.OCCLUDED:
            # From OCCLUDED: Normal recovery to any state
            if self.ground_streak >= OCCLUDED_TO_ANY_THRESHOLD:
                self.ball_state = BallState.ON_GROUND
            elif self.air_streak >= OCCLUDED_TO_ANY_THRESHOLD:
                self.ball_state = BallState.IN_AIR

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

                # Use the unified hierarchical tracker
                measurement_abs, annotation_sv, annotation_label = self.unified_tracker.get_measurement(
                    filtered_ball_detections, predicted_pos_abs, self.track_initialized,
                    coord_transform, gray_frame, frame_count
                )


                # Handle occlusion case
                if self.ball_state == BallState.OCCLUDED and annotation_sv is None: 
                    occluding_player = find_nearby_player(predicted_pos_abs, player_detections, OCCLUSION_ENTRY_THRESHOLD * 1.5)
                    if occluding_player: 
                        measurement_abs = occluding_player.get_anchors_coordinates(sv.Position.CENTER)[0]

                # Update state transitions - ALWAYS call this to detect occlusion
                self._manage_state_transitions(measurement_abs, annotation_sv, predicted_pos_abs, player_detections, pitch_mask, coord_transform)

                self.update_filter_for_state(self.ball_state)

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
                    
                    # Write to prediction file
                    if annotation_sv is not None: 
                        pred_file.write(f"{frame_count},-1,{annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][1]},{annotation_sv.xyxy[0][2]-annotation_sv.xyxy[0][0]},{annotation_sv.xyxy[0][3]-annotation_sv.xyxy[0][1]},1,-1,-1,-1\n")
                        state_num = self.ball_state.value
                        states_file.write(f"{frame_count},{state_num}\n")

                elif self.track_initialized:
                    self.lost_frames_count += 1
                    timeout = MAX_OCCLUSION_FRAMES if self.ball_state == BallState.OCCLUDED else MAX_LOST_FRAMES
                    if self.lost_frames_count > timeout:
                        self.track_initialized = False
                        # Also reset the unified tracker when main tracking is lost
                        self.unified_tracker._full_reset(frame_count)

                # Visualization - Don't show anything when in LOST mode
                if (self.track_initialized and annotation_sv is not None and 
                    self.unified_tracker.mode != TrackingMode.LOST):
                    # Add tracking mode information to the label
                    mode_info = f"[{self.unified_tracker.mode.name}]"
                    label = f"{annotation_label} {mode_info} ({self.ball_state.name})"
                    
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=annotation_sv, labels=[label])
                    self.box_annotator.color = sv.Color.YELLOW if self.ball_state == BallState.OCCLUDED else sv.Color.GREEN
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=annotation_sv)
                
                # Don't show predicted position when in LOST mode
                # (Removed the elif block that showed red predicted positions)

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
                print(f"Frame {frame_count}: State={self.ball_state.name}, measurement_abs={'Yes' if measurement_abs is not None else 'No'}")
                if predicted_pos_abs is not None:
                    nearby_player = find_nearby_player(predicted_pos_abs, player_detections, OCCLUSION_ENTRY_THRESHOLD)
                    if nearby_player is not None:
                        print(f"Frame {frame_count}: Ball near player! Distance < {OCCLUSION_ENTRY_THRESHOLD}")
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