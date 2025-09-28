import argparse
import sys
from enum import Enum
from collections import defaultdict
import numpy as np

# Define the ball states to map numbers back to readable names
class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    OCCLUDED = 3

def load_states_file(file_path):
    """Loads a state file (frame_id,state_id) into a dictionary."""
    labels = {}
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    frame_id_str, state_id_str = line.strip().split(',')
                    frame_id = int(frame_id_str)
                    state_id = int(state_id_str)
                    if state_id not in [s.value for s in BallState]:
                        print(f"Warning: Invalid state ID '{state_id}' on line {line_num} in {file_path}. Skipping.")
                        continue
                    labels[frame_id] = state_id
                except ValueError:
                    print(f"Warning: Skipping malformed line {line_num} in {file_path}: '{line.strip()}'")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)
    return labels

def main(args):
    """
    Compares predicted ball states against a ground truth file and reports accuracy.
    """
    print("Loading ground truth and prediction state files...")
    gt_states = load_states_file(args.gt)
    pred_states = load_states_file(args.pt)

    if not gt_states:
        print("Error: Ground truth file is empty or could not be read. Exiting.")
        sys.exit(1)

    # Initialize counters and a confusion matrix
    # Matrix shape is 3x3 for the 3 states. Rows = Actual, Cols = Predicted
    confusion_matrix = np.zeros((3, 3), dtype=int)
    total_frames = 0
    correct_predictions = 0

    # Iterate through the ground truth frames to ensure we evaluate every labeled frame
    for frame_id, actual_state_val in gt_states.items():
        total_frames += 1
        predicted_state_val = pred_states.get(frame_id)

        if predicted_state_val is None:
            # The model did not provide a prediction for a frame that has a ground truth.
            # This is a miss, but for state accuracy, we'll note it and continue.
            # In a confusion matrix, this is often handled separately or ignored.
            # Here, we will just count it as incorrect for the overall accuracy.
            continue
        
        # Adjust for 0-based indexing of the matrix
        actual_idx = actual_state_val - 1
        predicted_idx = predicted_state_val - 1
        
        confusion_matrix[actual_idx, predicted_idx] += 1
        
        if actual_state_val == predicted_state_val:
            correct_predictions += 1

    # --- Print Results ---
    print("\n--- Ball State Classification Report ---")

    # 1. Overall Accuracy
    overall_accuracy = (correct_predictions / total_frames) if total_frames > 0 else 0
    print(f"\nOverall Accuracy: {correct_predictions} / {total_frames} frames correct ({overall_accuracy:.2%})")

    # 2. Per-Class Accuracy
    print("\n--- Accuracy by State ---")
    for state in BallState:
        state_idx = state.value - 1
        correct_class_preds = confusion_matrix[state_idx, state_idx]
        total_class_instances = np.sum(confusion_matrix[state_idx, :])
        class_accuracy = (correct_class_preds / total_class_instances) if total_class_instances > 0 else 0
        print(f"  {state.name:<10}: {class_accuracy:.2%} ({correct_class_preds}/{total_class_instances})")

    # 3. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    header = " " * 12 + " ".join([f"{s.name:<10}" for s in BallState])
    print(header)
    print(" " * 11 + "-" * 34)
    for i, state in enumerate(BallState):
        row_str = f"Actual {state.name:<6}|"
        for j in range(len(BallState)):
            row_str += f" {str(confusion_matrix[i, j]):<9}"
        print(row_str)
    print("\n(Rows are actual states, Columns are predicted states)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of a ball state prediction model.")
    parser.add_argument("--gt", default="states.txt", type=str, 
                        help="Path to the ground truth states.txt file.")
    parser.add_argument("--pt", default="states_ball-det.txt", type=str, 
                        help="Path to the predicted states.txt file from your model.")
    args = parser.parse_args()
    main(args)

    
