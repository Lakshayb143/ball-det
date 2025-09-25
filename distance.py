import argparse
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

def get_box_center(box):
    """Calculates the center coordinates of a bounding box."""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return np.array([center_x, center_y])

def load_gt_data(file_path):
    """
    Loads ground truth data, including the 'is_airborne' flag.
    Assumes the format: frame,x,y,w,h,...,is_airborne
    """
    data = defaultdict(list)
    try:
        df = pd.read_csv(file_path, header=None)
        for _, row in df.iterrows():
            frame_id = int(row[0])
            # Assuming the columns are: 0:frame, 1:x, 2:y, 3:w, 4:h, ..., -1:is_airborne
            x1, y1, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            is_airborne = int(row.iloc[-1]) # The last column is the airborne flag
            x2 = x1 + w
            y2 = y1 + h
            # Store box, dimensions, and the airborne flag
            data[frame_id].append([[x1, y1, x2, y2], w, h, is_airborne])
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        sys.exit(1)
    return data

def load_pred_data(file_path):
    """
    Loads prediction data.
    Handles the format: frame,tracklet_id,x,y,w,h,...
    """
    data = defaultdict(list)
    try:
        df = pd.read_csv(file_path, header=None)
        for _, row in df.iterrows():
            frame_id = int(row[0])
            # FIX: Read from the correct columns, skipping tracklet_id at index 1
            x1, y1, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([[x1, y1, x2, y2], w, h])
    except FileNotFoundError:
        print(f"Error: Predictions file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        sys.exit(1)
    return data

def calculate_and_print_metrics(title, tp, fp, fn):
    """Calculates and prints precision, recall, and F1-score."""
    print(f"--- {title} ---")
    # print(f"  ✅ True Positives (TP):    {tp}")
    # print(f"  ❌ False Positives (FP):   {fp}")
    # print(f"  👻 False Negatives (FN):   {fn}")
    
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
    
    # Process frames where a ground truth exists
    for frame_id in sorted(all_gt_frames):
        gt_box, gt_w, gt_h, is_airborne = gt_data_by_frame[frame_id][0]
        category = 'air' if is_airborne == 1 else 'ground'
        
        if frame_id in pred_data_by_frame:
            pred_box, _, _ = pred_data_by_frame[frame_id][0]
            
            distance_threshold = args.perimeter_multiplier * (2 * (gt_w + gt_h))
            gt_center = get_box_center(gt_box)
            pred_center = get_box_center(pred_box)
            distance = np.linalg.norm(gt_center - pred_center)
            
            if distance <= distance_threshold:
                stats[category]['tp'] += 1
            else:
                stats[category]['fp'] += 1
        else:
            stats[category]['fn'] += 1
            
    # Process frames where a prediction exists but no ground truth does
    # These are always False Positives. We assume they are "grounded" FPs
    # as it's the most common state.
    # for frame_id in sorted(all_pred_frames - all_gt_frames):
    #     stats['ground']['fp'] += 1

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

