"""
Enhanced MMDetection3D Inference Script with Evaluation Metrics
Standalone version with all required functions included.

Computes: IoU, Precision, Recall, Average Precision (AP), mAP, F1-Score
"""

import os
import argparse
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

try:
    from mmdet3d.apis import LidarDet3DInferencer
except ImportError:
    print("Error: mmdetection3d not installed")
    exit()


def load_kitti_gt_labels(label_file):
    """
    Load KITTI ground truth labels from a .txt file.
    Returns list of 7D bounding boxes [x, y, z, l, w, h, yaw]
    """
    if not os.path.exists(label_file):
        return []
    
    gt_bboxes = []
    gt_labels = []
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            obj_type = parts[0]
            # Map KITTI classes to indices
            class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
            if obj_type not in class_map:
                continue
            
            # Extract dimensions (in camera coordinates)
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            yaw_cam = float(parts[14])
            
            # Convert camera coords to LiDAR coords
            # KITTI: x_lidar = z_cam, y_lidar = -x_cam, z_lidar = -y_cam
            x_lidar = z_cam
            y_lidar = -x_cam
            z_lidar = -y_cam
            yaw_lidar = -yaw_cam - (np.pi / 2.0)
            
            gt_bboxes.append([x_lidar, y_lidar, z_lidar, l, w, h, yaw_lidar])
            gt_labels.append(class_map[obj_type])
    
    return gt_bboxes, gt_labels


def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two bounding boxes.
    Boxes format: [x, y, z, l, w, h, yaw]
    """
    try:
        import torch
        from mmdet3d.structures import LiDARInstance3DBoxes
        from mmdet3d.structures.ops import box3d_overlap
        
        box1_tensor = torch.tensor([box1], dtype=torch.float32)
        box2_tensor = torch.tensor([box2], dtype=torch.float32)
        
        box1_obj = LiDARInstance3DBoxes(box1_tensor)
        box2_obj = LiDARInstance3DBoxes(box2_tensor)
        
        # Use built-in overlap function
        iou = box3d_overlap(box1_obj.tensor, box2_obj.tensor, mode='iou')
        return float(iou[0, 0])
    except Exception as e:
        # Fallback to BEV IoU
        return compute_bev_iou(box1, box2)


def compute_bev_iou(box1, box2):
    """
    Compute Bird's Eye View IoU as fallback.
    """
    try:
        from shapely.geometry import Polygon
        from shapely import affinity
        
        x1, y1, l1, w1, yaw1 = box1[0], box1[1], box1[3], box1[4], box1[6]
        x2, y2, l2, w2, yaw2 = box2[0], box2[1], box2[3], box2[4], box2[6]
        
        # Create rectangles centered at origin
        rect1 = Polygon([(-l1/2, -w1/2), (l1/2, -w1/2), (l1/2, w1/2), (-l1/2, w1/2)])
        rect2 = Polygon([(-l2/2, -w2/2), (l2/2, -w2/2), (l2/2, w2/2), (-l2/2, w2/2)])
        
        # Rotate
        rect1 = affinity.rotate(rect1, np.degrees(yaw1), origin=(0, 0))
        rect2 = affinity.rotate(rect2, np.degrees(yaw2), origin=(0, 0))
        
        # Translate
        rect1 = affinity.translate(rect1, xoff=x1, yoff=y1)
        rect2 = affinity.translate(rect2, xoff=x2, yoff=y2)
        
        intersection = rect1.intersection(rect2).area
        union = rect1.union(rect2).area
        
        return intersection / union if union > 0 else 0.0
    except:
        # Simple box overlap as last resort
        return 0.0


def match_predictions_to_gt(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.7):
    """
    Match predictions to ground truth using greedy algorithm.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            # Only match if classes are compatible
            if pred_labels[i] == gt_labels[j]:
                iou_matrix[i, j] = compute_iou_3d(pred_box, gt_box)
    
    # Greedy matching
    matches = []
    unmatched_preds = set(range(len(pred_boxes)))
    unmatched_gts = set(range(len(gt_boxes)))
    
    while unmatched_preds and unmatched_gts:
        max_iou = 0
        max_pred, max_gt = -1, -1
        
        for i in unmatched_preds:
            for j in unmatched_gts:
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_pred, max_gt = i, j
        
        if max_iou >= iou_threshold:
            matches.append((max_pred, max_gt, max_iou))
            unmatched_preds.remove(max_pred)
            unmatched_gts.remove(max_gt)
        else:
            break
    
    return matches, list(unmatched_preds), list(unmatched_gts)


def compute_precision_recall(matches, num_preds, num_gts):
    """Compute precision and recall."""
    tp = len(matches)
    fp = num_preds - tp
    fn = num_gts - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall, tp, fp, fn


def compute_average_precision(pred_scores, pred_matched, num_gts):
    """
    Compute Average Precision using 11-point interpolation.
    """
    if num_gts == 0:
        return 0.0
    
    # Sort by confidence descending
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_matched_sorted = np.array(pred_matched)[sorted_indices]
    
    # Compute precision-recall curve
    tp = np.cumsum(pred_matched_sorted)
    fp = np.cumsum(~pred_matched_sorted)
    recalls = tp / num_gts
    precisions = tp / (tp + fp)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.any(recalls >= t):
            ap += np.max(precisions[recalls >= t]) / 11.0
    
    return ap


def evaluate_sample(pred_dict, gt_bboxes, gt_labels_list, iou_threshold=0.7):
    """
    Evaluate predictions for a single sample.
    """
    # Extract predictions
    pred_bboxes = pred_dict.get('bboxes_3d', [])
    pred_scores = pred_dict.get('scores_3d', [])
    pred_labels = pred_dict.get('labels_3d', [])
    
    # Convert to numpy arrays
    if hasattr(pred_bboxes, 'tensor'):
        pred_bboxes = pred_bboxes.tensor.cpu().numpy()
    else:
        pred_bboxes = np.array(pred_bboxes) if len(pred_bboxes) > 0 else np.array([])
    
    if isinstance(pred_scores, list):
        pred_scores = np.array(pred_scores) if len(pred_scores) > 0 else np.array([])
    if isinstance(pred_labels, list):
        pred_labels = np.array(pred_labels) if len(pred_labels) > 0 else np.array([])
    
    gt_bboxes = np.array(gt_bboxes) if len(gt_bboxes) > 0 else np.array([])
    gt_labels = np.array(gt_labels_list) if len(gt_labels_list) > 0 else np.array([])
    
    # Match predictions to GT
    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels, iou_threshold
        )
    else:
        matches = []
        unmatched_preds = list(range(len(pred_bboxes)))
        unmatched_gts = list(range(len(gt_bboxes)))
    
    # Compute metrics
    precision, recall, tp, fp, fn = compute_precision_recall(matches, len(pred_bboxes), len(gt_bboxes))
    
    # Create matched array for AP
    pred_matched = np.zeros(len(pred_bboxes), dtype=bool)
    for pred_idx, _, _ in matches:
        pred_matched[pred_idx] = True
    
    ap = compute_average_precision(pred_scores, pred_matched, len(gt_bboxes)) if len(pred_scores) > 0 else 0.0
    avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'ap': ap,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'avg_iou': avg_iou,
        'num_preds': len(pred_bboxes),
        'num_gts': len(gt_bboxes),
        'num_matched': len(matches)
    }


def aggregate_metrics(all_metrics):
    """Aggregate metrics across samples."""
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_ap = np.mean([m['ap'] for m in all_metrics]) if all_metrics else 0.0
    mean_iou = np.mean([m['avg_iou'] for m in all_metrics if m['avg_iou'] > 0]) if all_metrics else 0.0
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'mAP': mean_ap,
        'mean_IoU': mean_iou,
        'f1_score': 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0,
        'total_TP': total_tp,
        'total_FP': total_fp,
        'total_FN': total_fn,
        'total_predictions': sum(m['num_preds'] for m in all_metrics),
        'total_ground_truth': sum(m['num_gts'] for m in all_metrics),
        'num_samples': len(all_metrics)
    }


def print_metrics_table(aggregated):
    """Print formatted metrics table."""
    print(" EVALUATION METRICS SUMMARY")
    print(f"Total Samples:        {aggregated['num_samples']}")
    print(f"Total Predictions:    {aggregated['total_predictions']}")
    print(f"Total Ground Truth:   {aggregated['total_ground_truth']}")
    print(f"True Positives (TP):  {aggregated['total_TP']}")
    print(f"False Positives (FP): {aggregated['total_FP']}")
    print(f"False Negatives (FN): {aggregated['total_FN']}")
    print(f"Precision:            {aggregated['precision']:.4f}")
    print(f"Recall:               {aggregated['recall']:.4f}")
    print(f"F1-Score:             {aggregated['f1_score']:.4f}")
    print(f"mAP (mean AP):        {aggregated['mAP']:.4f}")
    print(f"Mean IoU:             {aggregated['mean_IoU']:.4f}")


def main(args):
    print("Enhanced MMDetection3D Inference with Metrics Evaluation")
    print("="*80)
    
    # Initialize inferencer
    print(f"Loading model...")
    inferencer = LidarDet3DInferencer(
        args.model,
        args.checkpoint,
        device=args.device
    )
    
    # Build file list
    frames = args.frames.split(',') if args.frames != '-1' else []
    velodyne_dir = Path(args.input_path) / 'velodyne_reduced'
    label_dir = Path(args.input_path) / 'label_2'
    
    if args.frames == '-1':
        # Process all files in velodyne_reduced
        frames = sorted([f.stem for f in velodyne_dir.glob('*.bin')])
    
    print(f"Processing {len(frames)} samples...")
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Run inference and evaluation
    all_metrics = []
    
    for idx, frame_id in enumerate(frames):
        lidar_file = velodyne_dir / f"{frame_id}.bin"
        label_file = label_dir / f"{frame_id}.txt"
        
        if not lidar_file.exists():
            print(f"Warning: {lidar_file} not found, skipping")
            continue
        
        print(f"\n[{idx+1}/{len(frames)}] Processing: {frame_id}")
        
        # Run inference  
        result = inferencer({'points': str(lidar_file)})
        # Extract predictions - inferencer returns dict with 'predictions' key
        if isinstance(result, dict) and 'predictions' in result:
            pred_dict = result['predictions'][0]
        else:
            pred_dict = result[0] if isinstance(result, list) else result
        
        # Load ground truth
        gt_bboxes, gt_labels = [], []
        if label_file.exists():
            gt_bboxes, gt_labels = load_kitti_gt_labels(str(label_file))
        
        # Evaluate
        metrics = evaluate_sample(pred_dict, gt_bboxes, gt_labels, iou_threshold=args.iou_threshold)
        all_metrics.append(metrics)
        
        print(f"  Preds: {metrics['num_preds']}, GT: {metrics['num_gts']}, Matched: {metrics['num_matched']}")
        print(f"  P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, AP: {metrics['ap']:.3f}, IoU: {metrics['avg_iou']:.3f}")
    
    # Aggregate and print
    aggregated = aggregate_metrics(all_metrics)
    print_metrics_table(aggregated)
    
    # Save to JSON
    output_file = Path(args.out_dir) / 'evaluation_metrics.json'
    with open(output_file, 'w') as f:
        json.dump({
            'overall_metrics': aggregated,
            'per_sample_metrics': all_metrics
        }, f, indent=2)
    
    print(f"Metrics saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input-path', type=str, required=True, help="KITTI training folder")
    parser.add_argument('--frames', type=str, default='-1', help="Comma-separated frame IDs or -1 for all")
    parser.add_argument('--out-dir', type=str, default='./metrics_results')
    parser.add_argument('--iou-threshold', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    main(args)
