#!/usr/bin/env python3
"""Analyze MMDetection3D inference results"""
import json
import glob
from pathlib import Path

def analyze_predictions(json_path):
    """Parse a prediction JSON file and extract metrics"""
    with open(json_path) as f:
        data = json.load(f)
    
    # Count detections by class
    labels_3d = data.get('labels_3d', [])
    scores_3d = data.get('scores_3d', [])
    
    class_counts = {}
    class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
    
    for label in labels_3d:
        class_name = class_names.get(label, f'Class_{label}')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    avg_score = sum(scores_3d) / len(scores_3d) if scores_3d else 0
    
    return {
        "detections": len(labels_3d),
        "classes": class_counts,
        "avg_score": avg_score,
        "max_score": max(scores_3d) if scores_3d else 0,
        "min_score": min(scores_3d) if scores_3d else 0
    }

def main():
    results_dir = Path.home() / "Desktop" / "hw2_results"
    
    print("=" * 80)
    print("MMDetection3D Inference Results Analysis")
    print("=" * 80)
    
    # Analyze 1-class model
    print("\n### PointPillars (Car Only) Model ###\n")
    model1_dir = results_dir / "demo_results_pointpillars_hw2" / "preds"
    total_1class = 0
    for json_file in sorted(model1_dir.glob("*.json")):
        result = analyze_predictions(json_file)
        total_1class += result['detections']
        print(f"{json_file.stem}: {result['detections']} detections, "
              f"avg_score={result['avg_score']:.3f}, classes={result['classes']}")
    print(f"\nTotal detections (1-class): {total_1class}")
    print(f"Average per sample: {total_1class/4:.1f}")
    
    # Analyze 3-class model
    print("\n### PointPillars (3-Class: Car+Pedestrian+Cyclist) Model ###\n")
    model2_dir = results_dir / "demo_results_pointpillars_3class_hw2" / "preds"
    total_3class = 0
    class_totals = {}
    for json_file in sorted(model2_dir.glob("*.json")):
        result = analyze_predictions(json_file)
        total_3class += result['detections']
        for cls, count in result['classes'].items():
            class_totals[cls] = class_totals.get(cls, 0) + count
        print(f"{json_file.stem}: {result['detections']} detections, "
              f"avg_score={result['avg_score']:.3f}, classes={result['classes']}")
    print(f"\nTotal detections (3-class): {total_3class}")
    print(f"Average per sample: {total_3class/4:.1f}")
    print(f"Class distribution: {class_totals}")
    
    # Summary comparison
    print("COMPARISON SUMMARY")
    print(f"Model 1 (Car only):       {total_1class} total detections ({total_1class/4:.1f} avg/sample)")
    print(f"Model 2 (3-class):        {total_3class} total detections ({total_3class/4:.1f} avg/sample)")
    print(f"Additional objects found: {total_3class - total_1class} (pedestrians + cyclists)")
    print("\n")

if __name__ == "__main__":
    main()
