import cv2
import torch
import argparse
import os
import supervision as sv
import numpy as np
from rfdetr import RFDETRMedium

def main(args):
    """
    Main function to run raw model inference on a directory of images and save detections.
    """
    print(f"Loading model from checkpoint: {args.model_path}")
    try:
        model = RFDETRMedium(pretrain_weights=args.model_path)
        model.optimize_for_inference()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Simplified class map
    BALL_CLASS_ID = 0
    class_map = {BALL_CLASS_ID: "ball"}
    print(f"Using hardcoded class map: {class_map}")

    # Get the list of image files
    try:
        image_files = sorted([
            os.path.join(args.image_dir, f) 
            for f in os.listdir(args.image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not image_files:
            raise IOError(f"No images found in directory: {args.image_dir}")
    except Exception as e:
        print(e)
        return

    # Get frame dimensions from the first image
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read the first image at {image_files[0]}")
        return
    frame_height, frame_width, _ = first_frame.shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {args.output_path}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("\nStarting inference...")
    
    # Open the prediction file for writing
    with open(args.prediction_path, "w") as pred_file:
        # Loop through the image files
        for frame_count, image_path in enumerate(image_files, 1):
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue

            detections = model.predict(frame, confidence=args.confidence)

            # Filter for only ball detections
            ball_mask = detections.class_id == BALL_CLASS_ID
            ball_detections = detections[ball_mask]

            # --- Write detections to prediction file ---
            for box in ball_detections.xyxy:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                line = f"{frame_count},-1,{x1},{y1},{width},{height},1,-1,-1,-1\n"
                pred_file.write(line)

            labels = [
                f"{class_map[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(ball_detections.confidence, ball_detections.class_id)
            ]

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=ball_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=ball_detections, labels=labels)

            out_writer.write(annotated_frame)

    out_writer.release()
    print(f"Processing complete. Predictions saved to: {args.prediction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run raw RF-DETR inference on an image directory.")
    parser.add_argument("--model_path", default="ball.pth", type=str, help="Path to the model checkpoint file.")
    parser.add_argument("--image_dir", default="img1", type=str, help="Path to the directory containing image frames.")
    parser.add_argument("--prediction_path", default="prediction.txt", type=str, help="Path to save the prediction output file.")
    parser.add_argument("--output_path", type=str, default="output_img1_detections.mp4", help="Path to save the output annotated video.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for the output video.")
    args = parser.parse_args()
    main(args)
