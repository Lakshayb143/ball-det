import argparse
import numpy as np
from collections import defaultdict

def calculate_iou(boxA, boxB):
    boxA = np.array(boxA)
    boxB = np.array(boxB)
    
    # Intersection
    inter_tl = np.maximum(boxA[:2], boxB[:2])  # top-left
    inter_br = np.minimum(boxA[2:], boxB[2:])  # bottom-right
    
    inter_wh = np.maximum(0, inter_br - inter_tl)
    inter_area = inter_wh[0] * inter_wh[1]
    
    # Union
    area_A = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_B = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = area_A + area_B - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def load_boxes(file_path, ball_track_id=-1):
    """
    Loads bounding boxes, filtering for a specific track_id if provided.
    """
    boxes = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])

            # --- FIX: Only process the line if it's the ball's track_id ---
            # if ball_track_id is not -1, we filter. Otherwise, we load all.
            if ball_track_id != -1 and track_id != ball_track_id:
                continue

            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            x2 = x1 + w
            y2 = y1 + h
            boxes[frame_id].append([x1, y1, x2, y2])
    return boxes

def main(args):
    print("Loading ground truth (for ball only) and predictions...")
    gt_boxes_by_frame = load_boxes(args.gt, args.ball_track_id)
    pred_boxes_by_frame = load_boxes(args.pt) 
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    all_frame_ids = sorted(gt_boxes_by_frame.keys())

    for frame_id in all_frame_ids:
        gt_boxes = gt_boxes_by_frame.get(frame_id, [])
        pred_boxes = pred_boxes_by_frame.get(frame_id, [])
        
        gt_box = gt_boxes[0] if gt_boxes else None
        pred_box = pred_boxes[0] if pred_boxes else None

        if gt_box and pred_box:
            iou = calculate_iou(gt_box, pred_box)
            if iou >= args.iou_threshold:
                total_tp += 1
            else:
                total_fp += 1
        elif gt_box and not pred_box:
            total_fn += 1
        

    print("\n--- Evaluation Results (IoU-Based) ---")
    print(f"IoU Threshold: {args.iou_threshold}")
    print("--------------------------")
    print(f"✅ True Positives (TP):    {total_tp}")
    print(f"❌ False Positives (FP):   {total_fp} (Outliers)")
    print(f"👻 False Negatives (FN):   {total_fn} (Missed Detections)")
    print("--------------------------")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"📊 Precision: {precision:.2%}")
    print(f"🎯 Recall:    {recall:.2%}")
    print(f"⚖️ F1-Score:  {f1_score:.2f}")
    print("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MOT tracker's performance.")
    parser.add_argument("--gt", default="snmot196_gt.txt", type=str, help="Path to the ground truth file (gt.txt).")
    parser.add_argument("--pt", default="predictions.txt", type=str, help="Path to the predictions file (predictions.txt).")
    parser.add_argument("--ball_track_id", type=int, default=16, help="The track ID for the ball from gameinfo.ini.")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="IoU threshold for a detection to be a True Positive.")
    args = parser.parse_args()
    main(args)
