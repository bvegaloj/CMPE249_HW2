#!/bin/bash
# Create demo video from screenshots

cd ~/Desktop/hw2_results/visualizations/screenshots

# Create a list file for ffmpeg to concatenate
cat > filelist.txt << EOF
# KITTI PointPillars 1-class (Car-only)
file 'kitti_pp_1class_front.png'
duration 2
file 'kitti_pp_1class_top.png'
duration 2
file 'kitti_pp_1class_side.png'
duration 2
file 'kitti_pp_1class_perspective.png'
duration 2

# KITTI PointPillars 3-class (Car, Pedestrian, Cyclist)
file 'kitti_pp_3class_front.png'
duration 2
file 'kitti_pp_3class_top.png'
duration 2
file 'kitti_pp_3class_side.png'
duration 2
file 'kitti_pp_3class_perspective.png'
duration 2

# NuScenes CenterPoint (10 classes)
file 'nuscenes_cp_front.png'
duration 2
file 'nuscenes_cp_top.png'
duration 2
file 'nuscenes_cp_side.png'
duration 2
file 'nuscenes_cp_perspective.png'
duration 2

# Repeat last frame to avoid ffmpeg warning
file 'nuscenes_cp_perspective.png'
duration 0.5
EOF

# Generate video at 1080p
ffmpeg -f concat -safe 0 -i filelist.txt -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" -r 30 -pix_fmt yuv420p -c:v libx264 -crf 23 ../hw2_demo_video.mp4 -y

echo "Demo video created: ~/Desktop/hw2_results/visualizations/hw2_demo_video.mp4"
