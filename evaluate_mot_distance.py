import argparse
import numpy as np
from collections import defaultdict

def get_box_center(box):
    """Calculates the center coordinates of a bounding box."""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return np.array([center_x, center_y])

def load_data(file_path, ball_track_id):
    """
    Loads bounding boxes and dimensions, filtering for a specific track_id.
    """
    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])

            # --- FIX: Only process the line if it's the ball's track_id ---
            if track_id != ball_track_id:
                continue

            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([[x1, y1, x2, y2], w, h])
    return data

def main(args):
    print("Loading ground truth (for ball only) and predictions...")
    gt_data_by_frame = load_data(args.gt, args.ball_track_id)
    # Predictions file doesn't have multiple track IDs, so no filtering needed
    pred_data_by_frame = load_data(args.pt, -1) # Use -1 since our pred file has it
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    all_frame_ids = sorted(gt_data_by_frame.keys())

    for frame_id in all_frame_ids:
        gt_list = gt_data_by_frame.get(frame_id, [])
        pred_list = pred_data_by_frame.get(frame_id, [])
        
        gt_info = gt_list[0] if gt_list else None
        pred_info = pred_list[0] if pred_list else None

        if gt_info and pred_info:
            gt_box, gt_w, gt_h = gt_info
            pred_box, _, _ = pred_info

            distance_threshold = args.perimeter_multiplier * (2 * (gt_w + gt_h))
            gt_center = get_box_center(gt_box)
            pred_center = get_box_center(pred_box)
            distance = np.linalg.norm(gt_center - pred_center)

            if distance <= distance_threshold:
                total_tp += 1
            else:
                total_fp += 1
        elif gt_info and not pred_info:
            total_fn += 1
        

    print("\n--- Evaluation Results (Distance-Based) ---")
    print(f"Threshold: Center distance <= {args.perimeter_multiplier} * (GT BBox Perimeter)")
    print("---------------------------------------------")
    print(f"✅ True Positives (TP):    {total_tp}")
    print(f"❌ False Positives (FP):   {total_fp} (Outliers)")
    print(f"👻 False Negatives (FN):   {total_fn} (Missed Detections)")
    print("---------------------------------------------")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"📊 Precision: {precision:.2%}")
    print(f"🎯 Recall:    {recall:.2%}")
    print(f"⚖️ F1-Score:  {f1_score:.2f}")
    print("---------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MOT tracker's performance using center-point distance.")
    parser.add_argument("--gt", default="snmot196_gt.txt", type=str, help="Path to the ground truth file (gt.txt).")
    parser.add_argument("--pt", default="predictions.txt", type=str, help="Path to the predictions file (predictions.txt).")
    parser.add_argument("--ball_track_id", type=int, default=16, help="The track ID for the ball from gameinfo.ini.")
    parser.add_argument("--perimeter_multiplier", type=float, default=1.0, help="Multiplier for the perimeter threshold.")
    args = parser.parse_args()
    main(args)
