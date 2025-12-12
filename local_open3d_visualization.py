"""
Local Open3D visualization script for viewing PLY point clouds with bounding boxes.
This script opens an interactive 3D viewer and captures screenshots from multiple angles.
"""

import open3d as o3d
import json
import numpy as np
from pathlib import Path
import argparse

def create_box_line_set(box_dict):
    """Create Open3D LineSet from box dictionary"""
    corners = np.array(box_dict['corners'])
    
    # Define the 12 edges of the bounding box
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Color by confidence (red for high, yellow for low)
    score = box_dict['score']
    color = [1, 1 - score, 0]  # Red to yellow gradient
    colors = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set, box_dict['label'], score

def visualize_sample(ply_file, boxes_file, screenshots_dir=None, sample_name="sample"):
    """Visualize point cloud with bounding boxes and optionally save screenshots"""
    
    # Load point cloud
    print(f"\nLoading {ply_file}...")
    pcd = o3d.io.read_point_cloud(str(ply_file))
    print(f"  Points: {len(pcd.points)}")
    
    # Load bounding boxes
    print(f"Loading {boxes_file}...")
    with open(boxes_file, 'r') as f:
        boxes = json.load(f)
    print(f"  Boxes: {len(boxes)}")
    
    # Create geometries
    geometries = [pcd]
    
    # Add bounding boxes
    for box in boxes:
        line_set, label, score = create_box_line_set(box)
        geometries.append(line_set)
        print(f"    {label}: {score:.3f}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Detection - {sample_name}", width=1920, height=1080)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Set camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.set_front([0, -1, -0.3])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    
    if screenshots_dir:
        screenshots_dir = Path(screenshots_dir)
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Define camera angles
        views = [
            {'name': 'front', 'front': [0, -1, -0.3], 'lookat': [0, 0, 0], 'up': [0, 0, 1], 'zoom': 0.4},
            {'name': 'top', 'front': [0, 0, -1], 'lookat': [0, 0, 0], 'up': [0, 1, 0], 'zoom': 0.3},
            {'name': 'side', 'front': [-1, 0, -0.2], 'lookat': [0, 0, 0], 'up': [0, 0, 1], 'zoom': 0.4},
            {'name': 'perspective', 'front': [1, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, 0, 1], 'zoom': 0.35}
        ]
        
        for view in views:
            ctr.set_front(view['front'])
            ctr.set_lookat(view['lookat'])
            ctr.set_up(view['up'])
            ctr.set_zoom(view['zoom'])
            
            vis.poll_events()
            vis.update_renderer()
            
            screenshot_file = screenshots_dir / f"{sample_name}_{view['name']}.png"
            vis.capture_screen_image(str(screenshot_file), do_render=True)
            print(f"  Saved screenshot: {screenshot_file}")
        
        vis.destroy_window()
        print(f"\nSaved {len(views)} screenshots to {screenshots_dir}")
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("Interactive Visualization Controls:")
        print("  - Left mouse button: Rotate")
        print("  - Right mouse button: Pan")
        print("  - Scroll wheel: Zoom")
        print("  - Q or ESC: Quit")
        print("="*80)
        vis.run()
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D point clouds with bounding boxes')
    parser.add_argument('--ply', type=str, required=True, help='Path to PLY file')
    parser.add_argument('--boxes', type=str, required=True, help='Path to boxes JSON file')
    parser.add_argument('--screenshots', type=str, help='Directory to save screenshots (optional)')
    parser.add_argument('--name', type=str, default='sample', help='Sample name for screenshot files')
    
    args = parser.parse_args()
    
    visualize_sample(args.ply, args.boxes, args.screenshots, args.name)

if __name__ == '__main__':
    main()
