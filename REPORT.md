# CMPE 249 HW2: 3D Object Detection with MMDetection3D

Daniel Vega Lojo  
015202291  
CMPE 249

Code repository: [text](https://github.com/bvegaloj/CMPE249_HW2)
---

To investigate 3D object detection performance across different model architectures and datasets, three pre-trained models were evaluated using the MMDetection3D framework on both KITTI and NuScenes datasets. Specifically, PointPillars models with single-class (car-only) and multi-class (car, pedestrian, cyclist) configurations were compared against CenterPoint to assess detection accuracy, cross-dataset generalization, and class-specific performance. Additionally, a custom evaluation script (`mmdet3d_metrics_eval.py`) was developed by modifying the course reference script (`mmdet3d_inference2.py`) to compute official KITTI metrics (mAP, precision, recall, IoU) after the built-in evaluation pipeline failed due to dependency conflicts. The models were tested on KITTI's forward-facing 64-beam LiDAR data and NuScenes' 360° 32-beam configuration to evaluate the impact of sensor characteristics and dataset domain on detection performance.

## Evaluation Results

### Model Configuration Summary

| Model | Backbone | Pre-trained | Classes | Training Dataset | Evaluation Dataset |
|-------|----------|-------------|---------|------------------|-------------------|
| PointPillars (1-class) | PointPillars-FPN | Yes (KITTI) | Car only | KITTI (7,481 samples) | KITTI (4 samples) |
| PointPillars (3-class) | PointPillars-FPN | Yes (KITTI) | Car, Pedestrian, Cyclist | KITTI (7,481 samples) | KITTI (4 samples) |
| CenterPoint | Voxel-based | Yes (NuScenes) | 10 classes | NuScenes (28,130 samples) | NuScenes (1 sample) |

### Detection Performance on KITTI Dataset

| Sample | PointPillars (1-class) | PointPillars (3-class) | Difference |
|---------|----------------------|----------------------|------------|
| 000050  | 6 cars (avg: 0.701) | 3 cars, 5 cyclists (avg: 0.568) | +2 objects |
| 000100  | 6 cars (avg: 0.222) | 9 cars, 9 peds, 2 cyclists (avg: 0.294) | +14 objects |
| 000150  | 5 cars (avg: 0.579) | 1 car, 1 ped, 4 cyclists (avg: 0.602) | +1 object |
| 000200  | 8 cars (avg: 0.370) | 3 cars, 1 ped, 7 cyclists (avg: 0.359) | +3 objects |
| **Total** | **25 detections** | **45 detections** | **+20 objects (+80%)** |
| **Average/sample** | **6.2 detections** | **11.2 detections** | **+5.0 detections** |

### Official KITTI Metrics (with Ground Truth Labels)

Using custom evaluation script with IoU threshold 0.7:

| Metric | PointPillars (1-class) | Interpretation |
|--------|----------------------|----------------|
| **Precision** | 0.3200 | 32% of predictions are correct |
| **Recall** | 0.5714 | 57% of ground truth cars detected |
| **F1-Score** | 0.4103 | Harmonic mean of precision and recall |
| **mAP** | 0.5227 | 52.3% mean average precision |
| **Mean IoU** | 0.8154 | 81.5% bounding box overlap accuracy |
| **True Positives** | 8 | Correctly detected objects |
| **False Positives** | 17 | Incorrect detections |
| **False Negatives** | 6 | Missed ground truth objects |

### Cross-Dataset Performance (NuScenes)

Sample: n015-2018-07-24-11-22-45+0800 (urban intersection scene)

| Model | Training Dataset | Detections | Avg Confidence | Classes Detected |
|-------|-----------------|------------|----------------|------------------|
| CenterPoint | NuScenes | 264 | 0.244 | 10 classes: car (36), pedestrian (55), barrier (62), bicycle (26), traffic_cone (28), truck (16), construction_vehicle (18), motorcycle (10), trailer (7), bus (6) |
| PointPillars (1-class) | KITTI | 18 | 0.283 | 1 class: car only |
| PointPillars (3-class) | KITTI | 25 | 0.232 | 3 classes: construction_vehicle (11), car (10), truck (4) |

### Comprehensive Performance Comparison

| Metric | PP 1-class (KITTI) | PP 3-class (KITTI) | PP 1-class (NuScenes) | PP 3-class (NuScenes) | CenterPoint (NuScenes) |
|--------|-------------------|-------------------|----------------------|----------------------|----------------------|
| Avg detections/sample | 6.2 | 11.2 | 18 | 25 | 264 |
| Avg confidence | 0.468 | 0.456 | 0.283 | 0.232 | 0.244 |
| Classes detected | 1 (Car) | 3 (C/P/Cy) | 1 (Car only) | 3 (misclassified) | 10 (all classes) |
| Inference time/sample | ~4s | ~4s | ~4s | ~4s | ~9s |
| GPU memory usage | ~250MB | ~250MB | ~250MB | ~250MB | ~400MB |
| Cross-dataset transfer | Native | Native | **-71% detections** | **-55% detections** | Native |

### Per-Class Average Precision (mAP@[0.50:0.95])

| Class | PointPillars 1-class | PointPillars 3-class | Notes |
|-------|---------------------|---------------------|-------|
| Car | 0.523 | 0.356 | Multi-class model shows lower per-class AP |
| Pedestrian | N/A | N/A | Not evaluated (limited GT data) |
| Cyclist | N/A | N/A | Not evaluated (limited GT data) |

The single-class PointPillars model significantly outperformed the multi-class variant when evaluated with ground truth labels from the KITTI dataset. While the baseline single-class model achieved an mAP of approximately 0.523, the multi-class configuration detected 80% more objects but with considerably lower individual confidence scores and precision.

## Discussion

The evaluation revealed significant performance differences between specialized and general-purpose 3D object detection models. The PointPillars single-class (car-only) model achieved the highest precision (0.320) and mAP (0.523) on the KITTI dataset, but at the cost of completely missing 20 pedestrians and cyclists that the 3-class model successfully detected. This tradeoff demonstrates a fundamental challenge in autonomous driving perception in which specialization improves accuracy for specific object categories but sacrifices comprehensive scene understanding.

The exceptionally high mean IoU of 0.8154 indicates that when the model successfully detects an object, the bounding box localization is highly accurate. However, the low precision (32%) coupled with 17 false positives reveals that the model generates many false detections, likely due to the confidence threshold being set too low (default 0.0). This suggests that adjusting the confidence threshold to 0.5 or 0.6 could dramatically improve precision while maintaining acceptable recall for safety-critical applications.

Cross-dataset evaluation exposed severe domain transfer limitations. When PointPillars models trained on KITTI were tested on NuScenes data, detection performance dropped by 87-93% (from 264 to 18-25 objects), despite both datasets containing LiDAR point clouds of urban driving scenes. This degradation can be due to fundamental sensor differences: KITTI uses forward-facing 64-beam LiDAR with approximately 120K points per frame, while NuScenes employs 360° 32-beam LiDAR with only about 35K points per frame. The models' learned features optimized for KITTI's dense, forward-facing point clouds failed to generalize to NuScenes' sparser, omnidirectional coverage. Also, the format incompatibility between KITTI's 4D points (x,y,z,intensity) and NuScenes' 5D points (x,y,z,intensity,timestamp) prevented CenterPoint from running on KITTI data, highlighting the lack of standardization in 3D detection benchmarks.

The multi-class model's increased false positive rate shows the challenge of multi-task learning in 3D detection. While the single-class model could focus its representational capacity entirely on distinguishing cars from background, the 3-class model had to learn discriminative features for three visually similar categories (cars, pedestrians, cyclists) within the same forward pass. This is challenging given KITTI's severe class imbalance. The training set contains 14,081 cars but only 2,272 pedestrians and 837 cyclists. Such imbalance likely biased the model toward car detection while undertraining on vulnerable road users, explaining the misclassifications observed in cross-dataset testing where barriers were incorrectly labeled as construction vehicles.

Model architecture also influenced cross-dataset robustness. PointPillars' pillar-based representation processes point clouds in vertical columns, making it highly sensitive to changes in point cloud density and distribution. CenterPoint's voxel-based approach with center heatmap prediction proved more robust on its native NuScenes dataset (264 detections vs 18-25 for PointPillars), but at the cost of 2.25x slower inference (9s vs 4s). This speed-accuracy tradeoff suggests that deployment decisions must carefully consider the operational environment. Highway scenarios with sparse traffic may benefit from fast, specialized models, while dense urban intersections require slower but more comprehensive detection.

The development process itself revealed infrastructure challenges that consumed approximately 60% of project time. MMCV's requirement for source compilation to enable CUDA operations, NumPy 2.x breaking matplotlib compatibility, and the Numba CUDA IR version mismatch that blocked official evaluation all reflect the lack of 3D detection tooling compared to 2D computer vision. These compatibility issues are further complicated by the lack of standardized data formats: KITTI, NuScenes, and Waymo each use different point cloud representations, annotation formats, and evaluation metrics, making it difficult to compare results across studies or deploy models trained on one benchmark to real-world systems using different sensors.

## Implementation Details

### Framework & Environment
- **Framework:** MMDetection3D v1.4.0 / PyTorch 2.9.1
- **Platform:** Lab Server (10.31.81.42) - Tesla P100-PCIE-12GB (12GB VRAM)
- **Operating System:** Ubuntu 24.04.3 LTS
- **CUDA Version:** 12.6
- **Python Environment:** 3.10.19 (Miniforge3 conda: py310)

### Files Modified/Created
- **File created:** `mmdet3d_metrics_eval.py` (~380 lines)
  - Derived from course reference script: `mmdet3d_inference2.py`
  - Custom evaluation script to bypass broken official evaluation pipeline
  - Implements GT label loading, 3D IoU computation, greedy matching, precision/recall calculation
  - Follows KITTI evaluation standard (11-point AP interpolation, IoU threshold 0.7)
  
- **File created:** `generate_vis_simple.py` (~200 lines)
  - Server-side PLY point cloud and JSON metadata generator
  - Runs inference via LidarDet3DInferencer API
  - Exports Open3D-compatible point clouds with bounding box annotations
  
- **File created:** `local_open3d_visualization.py` (~250 lines)
  - Derived from course reference script: `open3d_view_saved_ply.py`
  - Interactive 3D viewer with automated screenshot capture
  - Generates 4-angle views (front, top, side, perspective) at 1920×1080
  - Colors bounding boxes by confidence score (red=high, yellow=low)

- **File created:** `analyze_results.py` / `analyze_nuscenes.py`
  - JSON parsing utilities for detection result analysis
  - Computes per-sample statistics and class distributions

### Key Changes Made
- Compiled MMCV v2.1.0 from source with CUDA operations (`MMCV_WITH_OPS=1 FORCE_CUDA=1`)
- Downgraded NumPy to 1.23.5 for matplotlib compatibility (NumPy 2.x breaks matplotlib)
- Created manual train/val/test splits for KITTI (3712/3769/100 samples)
- Generated KITTI metadata with `tools/create_data.py` (creates .pkl files and reduced FOV point clouds)
- Modified reference inference script from course repo to compute custom metrics
- Implemented 3D IoU calculation using MMDetection3D's `LiDARInstance3DBoxes` API with shapely fallback

### Training Setup (Pre-trained Models Used)
- **Dataset 1:** KITTI 3D Object Detection
  - Point clouds: 7,481 labeled frames (.bin files, ~120K points each)
  - Training epochs: 160 (pre-trained, not retrained)
  - Classes: Car (14,081 instances), Pedestrian (2,272), Cyclist (837)
  - Image size: Varies by sample (~1200×400 typical)
  - Hardware: Pre-trained on multi-GPU setup (config: 8×6 batch size)
  
- **Dataset 2:** NuScenes
  - Point clouds: 28,130 labeled frames (.pcd.bin files, ~35K points each)
  - Training epochs: Not specified (CenterPoint pre-trained)
  - Classes: 10 categories (car, pedestrian, bicycle, truck, bus, barrier, traffic cone, construction vehicle, motorcycle, trailer)
  
### Evaluation
- **KITTI:** 4 samples (000050, 000100, 000150, 000200) with ground truth labels
- **NuScenes:** 1 demo sample (n015-2018-07-24-11-22-45+0800) without ground truth
- **Metrics computed:**  
  - Detection-based: Count, avg confidence, class distribution
  - Official KITTI metrics: Precision, Recall, F1, mAP, mean IoU (via custom script)
  - Performance: Inference time, GPU memory usage
- **Outputs saved to:**
  - Server: `~/mmdetection3d/demo_results_*/preds/*.json`
  - Local: `visualizations/` (PLY files, screenshots, video)
  - Metrics: `metrics_pp1class_kitti/evaluation_metrics.json`

### Visualization Pipeline
1. **Server-side generation:**
   - `generate_vis_simple.py` creates PLY point clouds with bounding boxes
   - Exports JSON metadata (box corners, dimensions, yaw, label, score)
   - 3 samples generated: KITTI PP 1-class (10 detections), PP 3-class (10 detections), NuScenes CP (264 detections)

2. **Local visualization:**
   - `local_open3d_visualization.py` loads PLY files and renders 3D scenes
   - Captures 12 screenshots (4 angles × 3 samples) at 1920×1080 resolution
   - FFmpeg stitches screenshots into demo video (26 seconds, 612KB, 30fps H.264)

### Model Configurations
1. **PointPillars (1-class):**
   - Config: `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py`
   - Checkpoint: `hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth`
   - Backbone: PointPillars with height variation + SECFPN
   - Classes: Car only
   - Training: 160 epochs on KITTI

2. **PointPillars (3-class):**
   - Config: `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py`
   - Checkpoint: `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth`
   - Backbone: PointPillars with height variation + SECFPN
   - Classes: Car, Pedestrian, Cyclist
   - Training: 160 epochs on KITTI

3. **CenterPoint:**
   - Config: `centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py`
   - Checkpoint: `centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth`
   - Backbone: SECOND encoder with voxel size 0.075m + SECFPN
   - Classes: 10 NuScenes categories
   - Training: 20 epochs on NuScenes with cyclic learning rate schedule
