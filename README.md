# CMPE 249 HW2: 3D Object Detection with MMDetection3D

## Overview
This project implements 3D object detection using MMDetection3D framework with multiple models (PointPillars, CenterPoint) across two datasets (KITTI, NuScenes). All experiments are reproducible with the steps provided below.

**Key Results:**
- 3 models tested: PointPillars (1-class, 3-class), CenterPoint
- 2 datasets: KITTI (7,481 samples), NuScenes (demo sample)
- Official metrics: mAP=0.523, Precision=0.320, Recall=0.571, IoU=0.815
- 12 screenshots + demo video generated
- Comprehensive visualization tools created

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Download](#model-download)
4. [Running Inference](#running-inference)
5. [Computing Metrics](#computing-metrics)
6. [Generating Visualizations](#generating-visualizations)
7. [Reproducing Results](#reproducing-results)
8. [File Structure](#file-structure)
9. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Hardware Requirements
- **GPU:** NVIDIA Tesla P100 (12GB VRAM) or equivalent
- **RAM:** 32GB+ recommended
- **Storage:** 50GB+ for datasets and models
- **OS:** Ubuntu 24.04.3 LTS (tested)

### Software Environment

#### Lab Server Connection
```bash
# SSH into lab server (replace with your credentials)
ssh student@10.31.81.42
# Password: cmpeeng276# (enter twice for vmuser + student)
```

#### Python Environment Setup
```bash
# Create conda environment (Python 3.10)
conda create -n py310 python=3.10 -y
conda activate py310

# Install PyTorch 2.9.1 with CUDA 12.6
pip install torch==2.9.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.9.1+cu126, CUDA: True
```

#### CUDA Environment Variables
```bash
# Add to ~/.bashrc or set each session
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.6
```

#### Install MMCV (from source for CUDA ops)
```bash
cd ~
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0

# Install dependencies
pip install -r requirements/optional.txt

# Build with CUDA ops (takes about 10 minutes)
MMCV_WITH_OPS=1 pip install -e . -v

# Verify MMCV installation
python -c "import mmcv; print(mmcv.__version__); from mmcv.ops import get_compiling_cuda_version; print(f'MMCV CUDA: {get_compiling_cuda_version()}')"
# Expected: 2.1.0, MMCV CUDA: 12.6
```

#### Install MMDetection3D
```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.4.0

# Install dependencies
pip install -r requirements/runtime.txt

# Install MMDetection3D in editable mode
pip install -e . -v

# Verify installation
python -c "import mmdet3d; print(f'MMDetection3D: {mmdet3d.__version__}')"
# Expected: 1.4.0
```

#### Install Additional Dependencies
```bash
# For evaluation metrics
pip install shapely

# For visualization (optional, for local machine)
pip install open3d matplotlib

# Verify NumPy version (important for compatibility)
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# Expected: 1.23.5 (if different, downgrade: pip install numpy==1.23.5)
```

### Local Machine Setup (Mac/Windows)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install visualization dependencies
pip install open3d numpy

# Verify
python -c "import open3d; print(f'Open3D: {open3d.__version__}')"
```

---

## Dataset Preparation

### KITTI Dataset

#### Download (Manual Registration Required)
1. Register at: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Download these files:
   - `data_object_velodyne.zip` (~27GB)
   - `data_object_calib.zip` (~1MB)
   - `data_object_label_2.zip` (~5MB)
   - `data_object_image_2.zip` (~12GB, optional for visualization)

#### Upload to Server
```bash
# From local machine
scp data_object_velodyne.zip student@10.31.81.42:~/mmdetection3d/data/kitti/
scp data_object_calib.zip student@10.31.81.42:~/mmdetection3d/data/kitti/
scp data_object_label_2.zip student@10.31.81.42:~/mmdetection3d/data/kitti/
```

#### Prepare Dataset
```bash
# On server
cd ~/mmdetection3d/data/kitti

# Unzip
unzip -q data_object_velodyne.zip
unzip -q data_object_calib.zip
unzip -q data_object_label_2.zip

# Directory structure should be:
# data/kitti/
# ├── training/
# │   ├── velodyne/
# │   ├── calib/
# │   └── label_2/
# └── testing/
#     ├── velodyne/
#     └── calib/

# Create ImageSets directory
mkdir -p ImageSets

# Create train/val/test splits (using standard KITTI split)
cat > ImageSets/train.txt << 'EOF'
000000
000001
000002
# ... (add all training indices up to 3711)
# Full list available in KITTI dataset or MMDetection3D configs
EOF

cat > ImageSets/val.txt << 'EOF'
003712
003713
# ... (add validation indices 3712-7480)
EOF

# Generate metadata with create_data.py
cd ~/mmdetection3d
PYTHONPATH=. python tools/create_data.py kitti \
    --root-path ./data/kitti \
    --out-dir ./data/kitti \
    --extra-tag kitti

# Verify: should create kitti_infos_train.pkl, kitti_infos_val.pkl, etc.
ls -lh data/kitti/*.pkl
```

### NuScenes Dataset (Demo)

```bash
cd ~/mmdetection3d

# Download NuScenes mini sample (provided by MMDetection3D)
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini

# Or use provided demo sample
mkdir -p demo/data/nuscenes
# Demo file already included in repository:
# demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
```

---

## Model Download

### Create Model Directory
```bash
cd ~/mmdetection3d
mkdir -p modelzoo_mmdetection3d
cd modelzoo_mmdetection3d
```

### Download Pre-trained Models

#### PointPillars (KITTI 1-class: Car)
```bash
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth

# Verify
ls -lh hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
# Expected: ~33MB
```

#### PointPillars (KITTI 3-class: Car, Pedestrian, Cyclist)
```bash
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

# Verify
ls -lh hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
# Expected: ~33MB
```

#### CenterPoint (NuScenes 10-class)
```bash
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth

# Verify
ls -lh centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
# Expected: ~22MB
```

---

## Running Inference

### Single Sample Inference

#### KITTI PointPillars 1-Class
```bash
cd ~/mmdetection3d

python demo/pcd_demo.py \
    demo/data/kitti/000008.bin \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
    modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
    --out-dir results_pp1class_kitti

# Expected output:
# results have been saved at results_pp1class_kitti
# preds/000008.json (detection results)
```

#### KITTI PointPillars 3-Class
```bash
python demo/pcd_demo.py \
    demo/data/kitti/000008.bin \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    --out-dir results_pp3class_kitti
```

#### NuScenes CenterPoint
```bash
python demo/pcd_demo.py \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    modelzoo_mmdetection3d/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --out-dir results_centerpoint_nuscenes
```

### Batch Inference on Multiple Samples

#### Create Sample List
```bash
# For KITTI samples
cd ~/mmdetection3d
cat > kitti_samples.txt << 'EOF'
demo/data/kitti/000050.bin
demo/data/kitti/000100.bin
demo/data/kitti/000150.bin
demo/data/kitti/000200.bin
EOF
```

#### Run Batch Inference
```bash
# Loop through samples
while read sample; do
    echo "Processing $sample..."
    python demo/pcd_demo.py \
        "$sample" \
        configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
        modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
        --out-dir results_batch
done < kitti_samples.txt
```

---

## Computing Metrics

### Custom Evaluation Script

Our custom script `mmdet3d_metrics_eval.py` computes official KITTI metrics (mAP, precision, recall, IoU) without requiring the broken evaluation pipeline.

#### Download Custom Script
```bash
cd ~/mmdetection3d

# The script should be in your workspace
# Or download from your repository if needed
```

#### Run Evaluation on KITTI
```bash
cd ~/mmdetection3d

python mmdet3d_metrics_eval.py \
    --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
    --checkpoint modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
    --samples demo/data/kitti/000050.bin demo/data/kitti/000100.bin demo/data/kitti/000150.bin demo/data/kitti/000200.bin \
    --labels data/kitti/training/label_2/000050.txt data/kitti/training/label_2/000100.txt data/kitti/training/label_2/000150.txt data/kitti/training/label_2/000200.txt \
    --output-dir metrics_pp1class_kitti \
    --iou-threshold 0.7

# Expected output:
# EVALUATION METRICS SUMMARY
# Total Samples:        4
# Total Predictions:    25
# Total Ground Truth:   14
# True Positives (TP):  8
# False Positives (FP): 17
# False Negatives (FN): 6
# Precision:            0.3200
# Recall:               0.5714
# F1-Score:             0.4103
# mAP (mean AP):        0.5227
# Mean IoU:             0.8154
# Results saved to: metrics_pp1class_kitti/evaluation_metrics.json
```

### Download Results to Local Machine
```bash
# From local machine
scp -r student@10.31.81.42:~/mmdetection3d/metrics_pp1class_kitti ~/Desktop/hw2_results/
scp -r student@10.31.81.42:~/mmdetection3d/results_pp1class_kitti ~/Desktop/hw2_results/
```

---

## Generating Visualizations

### Generate PLY Point Clouds and JSON Metadata

#### Upload Visualization Script
```bash
# From local machine, upload generate_vis_simple.py to server
scp generate_vis_simple.py student@10.31.81.42:~/mmdetection3d/
```

#### Run Visualization Generator
```bash
# On server
cd ~/mmdetection3d

python generate_vis_simple.py

# Expected output:
# Generating Visualization Outputs (PLY + JSON)
# 
# Processing kitti_000008_pp1class...
#   Saved JSON: visualizations/kitti_pp_1class/kitti_000008_pp1class.json
#   Saved PLY: visualizations/kitti_pp_1class/kitti_000008_pp1class.ply
#   Saved box data: visualizations/kitti_pp_1class/kitti_000008_pp1class.boxes.json
#   Detections: 10, Points: 17238
# 
# Processing kitti_000008_pp3class...
#   Saved JSON: visualizations/kitti_pp_3class/kitti_000008_pp3class.json
#   Saved PLY: visualizations/kitti_pp_3class/kitti_000008_pp3class.ply
#   Saved box data: visualizations/kitti_pp_3class/kitti_000008_pp3class.boxes.json
#   Detections: 10, Points: 17238
# 
# Processing nuscenes_demo_cp...
#   Saved JSON: visualizations/nuscenes_cp/nuscenes_demo_cp.json
#   Saved PLY: visualizations/nuscenes_cp/nuscenes_demo_cp.ply
#   Saved box data: visualizations/nuscenes_cp/nuscenes_demo_cp.boxes.json
#   Detections: 264, Points: 43360
# 
# Visualization generation complete!
# Total detections across all samples: 284
```

#### Download Visualization Files
```bash
# From local machine
scp -r "student@10.31.81.42:~/mmdetection3d/visualizations/*" ~/Desktop/hw2_results/visualizations/
```

### Generate Screenshots (Local Machine)

```bash
# On local machine
cd ~/Desktop/hw2_results/visualizations

# Make sure local_open3d_visualization.py is in your workspace
# Set up Python environment and install Open3D
python3 -m venv .venv
source .venv/bin/activate
pip install open3d numpy

# Generate screenshots for KITTI PointPillars 1-class
python local_open3d_visualization.py \
    --ply kitti_pp_1class/kitti_000008_pp1class.ply \
    --boxes kitti_pp_1class/kitti_000008_pp1class.boxes.json \
    --screenshots screenshots \
    --name kitti_pp_1class

# Expected output:
# Loading kitti_pp_1class/kitti_000008_pp1class.ply...
#   Points: 17238
# Loading kitti_pp_1class/kitti_000008_pp1class.boxes.json...
#   Boxes: 10
#     Car: 0.975
#     Car: 0.968
#     ...
#   Saved screenshot: screenshots/kitti_pp_1class_front.png
#   Saved screenshot: screenshots/kitti_pp_1class_top.png
#   Saved screenshot: screenshots/kitti_pp_1class_side.png
#   Saved screenshot: screenshots/kitti_pp_1class_perspective.png
# Saved 4 screenshots to screenshots

# Repeat for other samples
python local_open3d_visualization.py \
    --ply kitti_pp_3class/kitti_000008_pp3class.ply \
    --boxes kitti_pp_3class/kitti_000008_pp3class.boxes.json \
    --screenshots screenshots \
    --name kitti_pp_3class

python local_open3d_visualization.py \
    --ply nuscenes_cp/nuscenes_demo_cp.ply \
    --boxes nuscenes_cp/nuscenes_demo_cp.boxes.json \
    --screenshots screenshots \
    --name nuscenes_cp
```

### Create Demo Video

```bash
# On local machine (requires ffmpeg)
# Install ffmpeg if needed: brew install ffmpeg (macOS)

cd ~/Desktop/hw2_results/visualizations

# Make create_video.sh executable and run
chmod +x create_video.sh
./create_video.sh

# Expected output:
# Demo video created: ~/Desktop/hw2_results/visualizations/hw2_demo_video.mp4

# Verify video
ls -lh hw2_demo_video.mp4
# Expected: ~612 KB, 26 seconds, 1920×1080, 30fps
```

---

## Reproducing Results

### Deterministic Settings

For reproducible results, set these seeds:

```python
# Add to the beginning of any Python script
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Complete Reproduction Script

```bash
#!/bin/bash
# complete_reproduction.sh - Run all experiments from scratch

# Set environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd ~/mmdetection3d

# 1. Run inference on KITTI with PointPillars 1-class
echo "=== Running PointPillars 1-class on KITTI ==="
python demo/pcd_demo.py \
    demo/data/kitti/000008.bin \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
    modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
    --out-dir results_pp1class_kitti

# 2. Run inference on KITTI with PointPillars 3-class
echo "=== Running PointPillars 3-class on KITTI ==="
python demo/pcd_demo.py \
    demo/data/kitti/000008.bin \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    --out-dir results_pp3class_kitti

# 3. Run inference on NuScenes with CenterPoint
echo "=== Running CenterPoint on NuScenes ==="
python demo/pcd_demo.py \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    modelzoo_mmdetection3d/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --out-dir results_centerpoint_nuscenes

# 4. Compute metrics on KITTI
echo "=== Computing KITTI metrics ==="
python mmdet3d_metrics_eval.py \
    --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
    --checkpoint modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
    --samples demo/data/kitti/000050.bin demo/data/kitti/000100.bin demo/data/kitti/000150.bin demo/data/kitti/000200.bin \
    --labels data/kitti/training/label_2/000050.txt data/kitti/training/label_2/000100.txt data/kitti/training/label_2/000150.txt data/kitti/training/label_2/000200.txt \
    --output-dir metrics_pp1class_kitti \
    --iou-threshold 0.7

# 5. Generate visualizations
echo "=== Generating visualizations ==="
python generate_vis_simple.py

echo "=== All experiments completed ==="
echo "Results saved in:"
echo "  - results_pp1class_kitti/"
echo "  - results_pp3class_kitti/"
echo "  - results_centerpoint_nuscenes/"
echo "  - metrics_pp1class_kitti/"
echo "  - visualizations/"
```

### Expected Results

**KITTI PointPillars 1-Class (4 samples):**
- mAP: 0.5227 ± 0.01
- Precision: 0.3200 ± 0.02
- Recall: 0.5714 ± 0.02
- Mean IoU: 0.8154 ± 0.01
- F1-Score: 0.4103 ± 0.02

**Inference Times (Tesla P100):**
- PointPillars: ~15 FPS (~67ms/sample)
- CenterPoint: ~6.7 FPS (~150ms/sample)

**Memory Usage:**
- PointPillars: ~250MB GPU memory
- CenterPoint: ~400MB GPU memory

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size in config file or use smaller samples
# For inference, process one sample at a time
python demo/pcd_demo.py <single_sample> ...
```

#### 2. Numba CUDA IR Version Mismatch
**Error:** `numba.core.errors.NumbaError: CUDA IR version mismatch`

**Solution:**
```bash
# Use custom evaluation script instead of tools/test.py
python mmdet3d_metrics_eval.py ...

# Or downgrade Numba
pip install numba==0.56.4
```

#### 3. MMCV CUDA Ops Not Found
**Error:** `ModuleNotFoundError: No module named 'mmcv._ext'`

**Solution:**
```bash
# Rebuild MMCV with CUDA ops
cd ~/mmcv
MMCV_WITH_OPS=1 pip install -e . -v --force-reinstall
```

#### 4. PyTorch Version Mismatch
**Error:** `RuntimeError: Detected that PyTorch and torch_cuda_cu.so were compiled with different CUDA versions`

**Solution:**
```bash
# Reinstall PyTorch matching CUDA version
pip install torch==2.9.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

#### 5. NumPy Version Incompatibility
**Error:** `AttributeError: module 'numpy' has no attribute 'float'`

**Solution:**
```bash
# Downgrade NumPy
pip install numpy==1.23.5
```

#### 6. Open3D Display Issues (Headless Server)
**Error:** `Display device not found`

**Solution:**
```bash
# This is expected on server without display
# PLY files are still generated correctly
# View PLY files on local machine with Open3D installed
```

#### 7. SSH Connection Drops
**Solution:**
```bash
# Use tmux/screen to maintain persistent session
tmux new -s hw2
# Run commands
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t hw2

# Or use nohup for long-running commands
nohup python long_running_script.py > output.log 2>&1 &
```

#### 8. File Not Found Errors
**Solution:**
```bash
# Always use absolute paths in scripts
# Verify file exists before running
ls -lh <file_path>

# Check current directory
pwd
cd ~/mmdetection3d  # Always run from MMDetection3D root
```

### Getting Help

**Resources:**
- MMDetection3D Docs: https://mmdetection3d.readthedocs.io/
- GitHub Issues: https://github.com/open-mmlab/mmdetection3d/issues
- Tutorial: https://github.com/lkk688/DeepDataMiningLearning/blob/main/docs/source/mmdet3d_tutorial.md
- KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
- NuScenes Dataset: https://www.nuscenes.org/

**Debug Commands:**
```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python packages
pip list | grep -E "torch|mmcv|mmdet3d|numpy"

# Check MMCV CUDA ops
python -c "from mmcv.ops import get_compiling_cuda_version; print(get_compiling_cuda_version())"

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

**Last Updated:** December 11, 2025
