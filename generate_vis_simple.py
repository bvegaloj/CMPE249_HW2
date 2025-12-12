"""
Generate visualization outputs for HW2 (no display required):
- PLY point clouds with predictions
- JSON metadata with box coordinates
"""

import os
import json
import numpy as np
from pathlib import Path
from mmdet3d.apis import LidarDet3DInferencer
import open3d as o3d

def save_point_cloud_with_boxes(points, boxes_3d, labels_3d, scores_3d, output_file, class_names=['Car']):
    """Save point cloud with bounding boxes as PLY file"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color point cloud by intensity if available
    if points.shape[1] >= 4:
        intensity = points[:, 3]
        colors = np.stack([intensity, intensity, intensity], axis=1)
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY
    o3d.io.write_point_cloud(str(output_file), pcd)
    
    # Save boxes separately as JSON (with visualization-ready format)
    boxes_list = []
    for i, (box, label, score) in enumerate(zip(boxes_3d, labels_3d, scores_3d)):
        # box format: [x, y, z, dx, dy, dz, yaw]
        center = box[:3].tolist()
        dims = box[3:6].tolist()
        yaw = float(box[6])
        
        # Create box corners for visualization
        l, w, h = dims
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners = np.vstack([x_corners, y_corners, z_corners])
        
        # Rotate
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        corners = rot_mat @ corners
        corners = (corners.T + box[:3]).tolist()
        
        box_dict = {
            'id': i,
            'center': center,
            'dimensions': dims,
            'yaw': yaw,
            'corners': corners,
            'label': class_names[int(label)] if int(label) < len(class_names) else f'class_{label}',
            'score': float(score)
        }
        boxes_list.append(box_dict)
    
    box_file = output_file.with_suffix('.boxes.json')
    with open(box_file, 'w') as f:
        json.dump(boxes_list, f, indent=2)
    
    return box_file

def generate_visualizations(lidar_file, config, checkpoint, output_dir, sample_name, class_names):
    """Generate PLY and JSON outputs for a sample"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inferencer
    print(f"\nProcessing {sample_name}...")
    inferencer = LidarDet3DInferencer(config, checkpoint, device='cuda:0')
    
    # Run inference
    result = inferencer({'points': str(lidar_file)}, no_save_vis=True, return_datasamples=True)
    
    # Extract results
    pred = result['predictions'][0]
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    # Access data from Det3DDataSample object
    boxes_3d = pred.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    labels_3d = pred.pred_instances_3d.labels_3d.cpu().numpy()
    scores_3d = pred.pred_instances_3d.scores_3d.cpu().numpy()
    
    # Save JSON metadata
    json_file = output_dir / f'{sample_name}.json'
    metadata = {
        'sample': sample_name,
        'lidar_file': str(lidar_file),
        'num_detections': len(boxes_3d),
        'num_points': len(points),
        'detections': []
    }
    
    for i, (box, label, score) in enumerate(zip(boxes_3d, labels_3d, scores_3d)):
        detection = {
            'id': i,
            'box_3d': box.tolist(),
            'label': class_names[int(label)] if int(label) < len(class_names) else f'class_{label}',
            'score': float(score)
        }
        metadata['detections'].append(detection)
    
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved JSON: {json_file}")
    
    # Save PLY point cloud
    ply_file = output_dir / f'{sample_name}.ply'
    box_file = save_point_cloud_with_boxes(points, boxes_3d, labels_3d, scores_3d, ply_file, class_names)
    print(f"  Saved PLY: {ply_file}")
    print(f"  Saved box data: {box_file}")
    
    print(f"  Detections: {len(boxes_3d)}, Points: {len(points)}")
    return len(boxes_3d)

def main():
    # KITTI class names
    KITTI_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    
    # NuScenes class names
    NUSCENES_CLASSES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    # Configuration
    samples = [
        {
            'lidar': 'demo/data/kitti/000008.bin',
            'config': 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py',
            'checkpoint': 'modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',
            'output_dir': 'visualizations/kitti_pp_1class',
            'name': 'kitti_000008_pp1class',
            'classes': ['Car']
        },
        {
            'lidar': 'demo/data/kitti/000008.bin',
            'config': 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
            'checkpoint': 'modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth',
            'output_dir': 'visualizations/kitti_pp_3class',
            'name': 'kitti_000008_pp3class',
            'classes': KITTI_CLASSES
        },
        {
            'lidar': 'demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin',
            'config': 'configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
            'checkpoint': 'modelzoo_mmdetection3d/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth',
            'output_dir': 'visualizations/nuscenes_cp',
            'name': 'nuscenes_demo_cp',
            'classes': NUSCENES_CLASSES
        }
    ]
    
    print("Generating Visualization Outputs (PLY + JSON)")
    
    total_detections = 0
    for sample in samples:
        try:
            detections = generate_visualizations(
                sample['lidar'],
                sample['config'],
                sample['checkpoint'],
                sample['output_dir'],
                sample['name'],
                sample['classes']
            )
            total_detections += detections
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Visualization generation complete!")
    print(f"Total detections across all samples: {total_detections}")
    print("\nGenerated files:")
    print("  - *.ply: Point cloud files (can be viewed with Open3D)")
    print("  - *.boxes.json: Bounding box coordinates and labels")
    print("  - *.json: Complete detection metadata")

if __name__ == '__main__':
    main()
