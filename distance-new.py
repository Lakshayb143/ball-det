import argparse
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

def get_box_center(box):
    """Calculates the center coordinates of a bounding box."""
    # box is [x1, y1, x2, y2]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return np.array([center_x, center_y])

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
            data[frame_id].append([[x1, y1, x2, y2], w, h, is_airborne])
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
            data[frame_id].append([[x1, y1, x2, y2], w, h])
    except FileNotFoundError:
        print(f"Error: Predictions file not found at '{file_path}'")
        sys.exit(1)
    return data

def calculate_and_print_metrics(title, tp, fp, fn):
    """Calculates and prints precision, recall, and F1-score."""
    print(f"--- {title} ---")
    print(f"  ✅ True Positives (TP):    {tp}")
    print(f"  ❌ False Positives (FP):   {fp}")
    print(f"  👻 False Negatives (FN):   {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  📊 Precision: {precision:.2%}")
    print(f"  🎯 Recall:    {recall:.2%}")
    print(f"  ⚖️ F1-Score:  {f1_score:.2f}")
    print("-" * (len(title) + 6))

def main(args):
    print("Loading labeled ground truth and predictions...")
    gt_data_by_frame = load_gt_data(args.gt)
    pred_data_by_frame = load_pred_data(args.pt)
    
    stats = {
        'air': {'tp': 0, 'fp': 0, 'fn': 0},
        'ground': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    all_gt_frames = set(gt_data_by_frame.keys())
    all_pred_frames = set(pred_data_by_frame.keys())
    
    # Iterate through all frames that have either a GT or a prediction
    all_frames = sorted(all_gt_frames.union(all_pred_frames))

    for frame_id in all_frames:
        gt_info = gt_data_by_frame.get(frame_id)
        pred_info = pred_data_by_frame.get(frame_id)
        
        # Case 1: Both GT and Prediction exist for the frame
        if gt_info and pred_info:
            gt_box, gt_w, gt_h, is_airborne = gt_info[0]
            pred_box, _, _ = pred_info[0]
            category = 'air' if is_airborne == 1 else 'ground'
            
            distance_threshold = args.perimeter_multiplier * (2 * (gt_w + gt_h))
            gt_center = get_box_center(gt_box)
            pred_center = get_box_center(pred_box)
            distance = np.linalg.norm(gt_center - pred_center)
            
            if distance <= distance_threshold:
                # Correct detection
                stats[category]['tp'] += 1
            else:
                # Incorrect detection: it's both a miss and a false alarm
                stats[category]['fp'] += 1
        
        # Case 2: GT exists, but no prediction was made
        elif gt_info and not pred_info:
            _, _, _, is_airborne = gt_info[0]
            category = 'air' if is_airborne == 1 else 'ground'
            # Missed detection
            stats[category]['fn'] += 1
            
        # Case 3: Prediction exists, but no GT was there
            #       elif not gt_info and pred_info:
            # False alarm. Assume it's a "ground" FP as it's the most common state.
        #stats['ground']['fp'] += 1

    total_tp = stats['air']['tp'] + stats['ground']['tp']
    total_fp = stats['air']['fp'] + stats['ground']['fp']
    total_fn = stats['air']['fn'] + stats['ground']['fn']
    
    print("\n--- Evaluation Results (Distance-Based) ---")
    print(f"Threshold: Center distance <= {args.perimeter_multiplier} * (GT BBox Perimeter)\n")
    
    calculate_and_print_metrics("Airborne Ball Performance", stats['air']['tp'], stats['air']['fp'], stats['air']['fn'])
    calculate_and_print_metrics("Grounded Ball Performance", stats['ground']['tp'], stats['ground']['fp'], stats['ground']['fn'])
    calculate_and_print_metrics("Overall Performance", total_tp, total_fp, total_fn)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ball detection, separating 'airborne' and 'grounded' performance.")
    parser.add_argument("--gt", default="ball_gt_labeled.txt", type=str, 
                        help="Path to the final, verified ground truth file with airborne labels.")
    parser.add_argument("--pt", default="predictions.txt", type=str, 
                        help="Path to the predictions file from your model.")
    parser.add_argument("--perimeter_multiplier", type=float, default=1.0, 
                        help="Multiplier for the perimeter threshold. Default is 1.0.")
    args = parser.parse_args()
    main(args)


