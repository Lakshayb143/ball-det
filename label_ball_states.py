import argparse
import cv2
import os
import sys
from enum import Enum

# Define the ball states as requested
class BallState(Enum):
    ON_GROUND = 1
    IN_AIR = 2
    OCCLUDED = 3

def load_existing_labels(file_path):
    """Loads existing labels from the output file to allow resuming."""
    labels = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    frame_id_str, state_id_str = line.strip().split(',')
                    labels[int(frame_id_str)] = int(state_id_str)
                except ValueError:
                    print(f"Warning: Skipping malformed line in existing label file: {line.strip()}")
    return labels

def save_labels(file_path, labels):
    """Saves all current labels to the file, sorted by frame number."""
    with open(file_path, 'w') as f:
        for frame_id in sorted(labels.keys()):
            f.write(f"{frame_id},{labels[frame_id]}\n")

def main(args):
    """
    An interactive tool to label the state of the ball in each frame of a video.
    """
    # --- Load Data and Prepare Files ---
    image_files = sorted(
        [f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda f: int(''.join(filter(str.isdigit, f))) # Natural sort for frame numbers
    )
    
    if not image_files:
        print(f"Error: No image files found in '{args.frames_dir}'")
        sys.exit(1)
        
    total_frames = len(image_files)
    labels = load_existing_labels(args.output_file)
    
    # Start from the last labeled frame + 1, or from the beginning
    current_frame_index = len(labels) if labels else 0

    print("\n--- Ball State Labeling Tool ---")
    print("  KEYS   | ACTION")
    print("  ---------------------------------")
    print("  1      | Label as ON_GROUND")
    print("  2      | Label as IN_AIR")
    print("  3      | Label as OCCLUDED")
    print("  ---------------------------------")
    print("  A / Left Arrow  | Previous Frame")
    print("  D / Right Arrow | Next Frame")
    print("  Q / ESC         | Quit and Save")
    print("---------------------------------\n")

    while current_frame_index < total_frames:
        frame_id = current_frame_index + 1
        image_path = os.path.join(args.frames_dir, image_files[current_frame_index])
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_id} at '{image_path}'")
            current_frame_index += 1
            continue

        # --- Display Information on Frame ---
        display_frame = frame.copy()
        
        # Display current status if already labeled
        current_label = labels.get(frame_id)
        status_text = "Status: NOT LABELED"
        if current_label is not None:
            try:
                status_text = f"Status: {BallState(current_label).name}"
            except ValueError:
                status_text = "Status: UNKNOWN LABEL"

        info_text = f"Frame: {frame_id} / {total_frames}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add instructions directly on the screen
        instructions = "Keys: [1] Ground [2] Air [3] Occluded | [A/D] Navigate | [Q] Quit"
        cv2.putText(display_frame, instructions, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Ball State Labeling Tool", display_frame)

        # --- Controls ---
        key = cv2.waitKey(0) & 0xFF

        # --- Navigation ---
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('d') or key == 83:  # 'd' or right arrow
            current_frame_index = min(current_frame_index + 1, total_frames - 1)
        elif key == ord('a') or key == 81:  # 'a' or left arrow
            current_frame_index = max(current_frame_index - 1, 0)
        
        # --- Labeling ---
        elif key in [ord('1'), ord('2'), ord('3')]:
            if key == ord('1'):
                labels[frame_id] = BallState.ON_GROUND.value
            elif key == ord('2'):
                labels[frame_id] = BallState.IN_AIR.value
            elif key == ord('3'):
                labels[frame_id] = BallState.OCCLUDED.value
            
            # Save progress immediately and move to the next frame
            save_labels(args.output_file, labels)
            print(f"Frame {frame_id} labeled as {BallState(labels[frame_id]).name}. Progress saved.")
            current_frame_index = min(current_frame_index + 1, total_frames - 1)

    print("\nLabeling complete or exited. Final labels saved.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An interactive tool to label the state of a ball in video frames.")
    parser.add_argument("--frames_dir", default="img1", type=str, 
                        help="Path to the directory containing the video frames (images).")
    parser.add_argument("--output_file", default="states.txt", type=str, 
                        help="Path to the output ground truth file. Default is states.txt.")
    args = parser.parse_args()
    main(args)

