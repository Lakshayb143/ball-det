import argparse
import cv2
import sys

def main(args):
    """
    A simple video player to view a video frame by frame for debugging.
    """
    # --- Open Video File ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{args.video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_pos = 0
    is_paused = True

    print("\n--- Video Viewer Controls ---")
    print("  SPACEBAR : Play/Pause")
    print("  A / Left Arrow :  Previous Frame")
    print("  D / Right Arrow : Next Frame")
    print("  Q : Quit")
    print("---------------------------\n")

    while True:
        # Set the video to the correct position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            current_frame_pos = total_frames - 1 # Stay on the last frame
            key = cv2.waitKey(0) # Wait for a key press before exiting
            if key in [ord('q'), 27]: # Quit on 'q' or ESC
                break
            else:
                continue

        # Frame IDs are 1-based, video position is 0-based
        frame_id = current_frame_pos + 1
        
        # Create a copy to draw on, so we can refresh text
        display_frame = frame.copy()
        
        # Display Frame Information
        info_text = f"Frame: {frame_id} / {total_frames}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Simple Video Viewer", display_frame)
        
        # --- Controls ---
        # If paused, wait indefinitely. If playing, wait for 30ms.
        wait_time = 0 if is_paused else 30 
        key = cv2.waitKey(wait_time) & 0xFF

        if key in [ord('q'), 27]:  # 'q' or ESC key
            break
        elif key == ord(' '):  # Spacebar
            is_paused = not is_paused
        elif key in [ord('d'), 83, 77]:  # 'd', Right Arrow (Linux=83, Windows=77)
            is_paused = True
            current_frame_pos = min(current_frame_pos + 1, total_frames - 1)
        elif key in [ord('a'), 81, 75]:  # 'a', Left Arrow (Linux=81, Windows=75)
            is_paused = True
            current_frame_pos = max(current_frame_pos - 1, 0)
        
        if not is_paused:
            current_frame_pos += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple frame-by-frame video viewer for debugging.")
    parser.add_argument("--video_path", required=True, type=str, 
                        help="Path to the video file to display.")
    args = parser.parse_args()
    main(args)


