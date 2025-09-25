import argparse
import numpy as np
import pandas as pd
import sys
import cv2
import os
from collections import defaultdict

def calculate_iou(box_a, box_b):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Assumes box format is [x1, y1, x2, y2].
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def load_gt_data(file_path):
    """
    Loads ground truth data, including the 'is_airborne' flag.
    Format: frame,x,y,w,h,...,is_airborne
    """
    data = defaultdict(list)
    try:
        df = pd.read_csv(file_path, header=None)
        for _, row in df.iterrows():
            frame_id = int(row[0])
            x1, y1, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            is_airborne = int(row.iloc[-1])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([[int(x1), int(y1), int(x2), int(y2)], is_airborne])
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{file_path}'")
        sys.exit(1)
    return data

def load_pred_data(file_path):
    """
    Loads prediction data. Format: frame,tracklet_id,x,y,w,h,...
    """
    data = defaultdict(list)
    try:
        df = pd.read_csv(file_path, header=None)
        for _, row in df.iterrows():
            frame_id = int(row[0])
            x1, y1, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([int(x1), int(y1), int(x2), int(y2)])
    except FileNotFoundError:
        print(f"Error: Predictions file not found at '{file_path}'")
        sys.exit(1)
    return data

def main(args):
    # --- Load Data ---
    gt_data = load_gt_data(args.gt)
    pred_data = load_pred_data(args.pt)
    
    # --- Open Video File ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{args.video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_pos = 0

    # --- Interactive Visualization Loop ---
    while current_frame_pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at position {current_frame_pos}")
            current_frame_pos += 1
            continue
        
        # Frame IDs in GT file are 1-based, video position is 0-based
        frame_id = current_frame_pos + 1
        
        # --- Get GT and Prediction for this frame ---
        gt_info = gt_data.get(frame_id)
        pred_box = pred_data.get(frame_id)
        
        status_text = ""
        iou_text = ""
        gt_status = ""

        # Draw Ground Truth Box (if it exists)
        if gt_info:
            gt_box, is_airborne = gt_info[0]
            gt_status = "Airborne" if is_airborne == 1 else "Grounded"
            cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Ground Truth", (gt_box[0], gt_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Prediction Box and determine status
        if pred_box:
            pred_box = pred_box[0]
            if gt_info:
                iou = calculate_iou(gt_box, pred_box)
                iou_text = f"IoU: {iou:.2f}"
                if iou >= args.iou_threshold:
                    status_text = "Status: True Positive"
                    cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (255, 0, 0), 2)
                    cv2.putText(frame, "Prediction (TP)", (pred_box[0], pred_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    status_text = "Status: False Positive"
                    cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "Prediction (FP)", (pred_box[0], pred_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                status_text = "Status: False Positive"
                cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Prediction (FP)", (pred_box[0], pred_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif gt_info:
            status_text = "Status: False Negative"
        
        # --- Display Information on Frame ---
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"GT: {gt_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, iou_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Detection Visualizer", frame)
        
        # --- Controls ---
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') or key == 83:  # 'd' or right arrow
            current_frame_pos = min(current_frame_pos + 1, total_frames - 1)
        elif key == ord('a') or key == 81:  # 'a' or left arrow
            current_frame_pos = max(current_frame_pos - 1, 0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and debug ball detection performance frame by frame.")
    parser.add_argument("--video_path", default="output_v9.mp4", type=str, 
                        help="Path to the video file to analyze.")
    parser.add_argument("--gt", default="196gta.txt", type=str, 
                        help="Path to the final, verified ground truth file with airborne labels.")
    parser.add_argument("--pt", default="predictions_v9_2.txt", type=str, 
                        help="Path to the predictions file from your model.")
    parser.add_argument("--iou_threshold", type=float, default=0.1, 
                        help="IoU threshold for a detection to be considered a True Positive.")
    args = parser.parse_args()
    main(args)


