import argparse
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

def calculate_iou(box_a, box_b):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Assumes box format is [x1, y1, x2, y2].
    """
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the union area
    union_area = float(box_a_area + box_b_area - inter_area)
    
    # Compute the IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

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
            x1, y1, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            is_airborne = int(row.iloc[-1])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([[x1, y1, x2, y2], is_airborne])
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        sys.exit(1)
    return data

def load_pred_data(file_path):
    """
    Loads prediction data. Handles the format: frame,tracklet_id,x,y,w,h,...
    """
    data = defaultdict(list)
    try:
        df = pd.read_csv(file_path, header=None)
        for _, row in df.iterrows():
            frame_id = int(row[0])
            x1, y1, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            x2 = x1 + w
            y2 = y1 + h
            data[frame_id].append([x1, y1, x2, y2])
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
    
    # Process frames where a ground truth exists
    for frame_id in sorted(all_gt_frames):
        gt_box, is_airborne = gt_data_by_frame[frame_id][0]
        category = 'air' if is_airborne == 1 else 'ground'
        
        if frame_id in pred_data_by_frame:
            pred_box = pred_data_by_frame[frame_id][0]
            
            iou = calculate_iou(gt_box, pred_box)
            
            if iou >= args.iou_threshold:
                stats[category]['tp'] += 1
            else:
                stats[category]['fp'] += 1
        else:
            stats[category]['fn'] += 1
            
    # Process frames where a prediction exists but no ground truth does
    # for frame_id in sorted(all_pred_frames - all_gt_frames):
    #     stats['ground']['fp'] += 1

    total_tp = stats['air']['tp'] + stats['ground']['tp']
    total_fp = stats['air']['fp'] + stats['ground']['fp']
    total_fn = stats['air']['fn'] + stats['ground']['fn']
    
    print("\n--- Evaluation Results (IoU-Based) ---")
    print(f"Threshold: IoU >= {args.iou_threshold}\n")
    
    calculate_and_print_metrics("Airborne Ball Performance", stats['air']['tp'], stats['air']['fp'], stats['air']['fn'])
    calculate_and_print_metrics("Grounded Ball Performance", stats['ground']['tp'], stats['ground']['fp'], stats['ground']['fn'])
    calculate_and_print_metrics("Overall Performance", total_tp, total_fp, total_fn)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ball detection using IoU, separating 'airborne' and 'grounded' performance.")
    parser.add_argument("--gt", default="ball_gt_labeled.txt", type=str, 
                        help="Path to the final, verified ground truth file with airborne labels.")
    parser.add_argument("--pt", default="predictions.txt", type=str, 
                        help="Path to the predictions file from your model.")
    parser.add_argument("--iou_threshold", type=float, default=0.1, 
                        help="IoU threshold for a detection to be considered a True Positive. Default is 0.5.")
    args = parser.parse_args()
    main(args)
