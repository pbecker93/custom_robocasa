# custom_robocasa

This repository contains the custom components for the Robocasa benchmark. The custom components
provide point cloud wrappers which can be used to generate point clouds from the Robocasa dataset and simulation.

## Installation

```
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa

sh install.sh
```

## Usage:
Main script to use is [custom_robocasa/custom_robocasa/robocasa/scripts/sync_dataset_states_to_obs.py](custom_robocasa/robocasa/scripts/sync_dataset_states_to_obs.py).
This script basically reruns all demonstrations in a robocasa dataset, rerenders the images and stores them according to a dataset structure `config.yaml`.

```bash
python sync_dataset_states_to_obs.py --dataset <path_to_some_robocasa_hdf5> --config <path_to_some_dataset_config_yaml> --camera_names robot0_agentview_left robot0_agentview_right --camera_width 1600 --camera_height 1200 --segmentation --num_procs=1
```

## Example config file:
```yaml
recordings_folder: recordings
demonstrations_folder: demonstrations
canonical_folder: canonical

#========= SUBFOLDERS ============
# Actions
action_subfolder: actions
simple_pc_action_subfolder: pc_actions

# Images
color_subfolder:
 - robot0_agentview_left_image
 - robot0_agentview_right_image
depth_subfolder:
 - robot0_agentview_left_depth
 - robot0_agentview_right_depth

# Binary Masks
binary_mask_target_subfolder: 
 - robot0_agentview_left_segmentation_mask/container
 - robot0_agentview_right_segmentation_mask/container
binary_mask_tool_subfolder:
 - robot0_agentview_left_segmentation_mask/obj
 - robot0_agentview_right_segmentation_mask/obj

# Some runs somehow have two possible segmentation keywords, so you can add them here, if this is not found it will be ignored
backup_binary_mask_target_subfolder: 
 - robot0_agentview_left_segmentation_mask/container_
 - robot0_agentview_right_segmentation_mask/container_

backup_binary_mask_tool_subfolder:
 - robot0_agentview_left_segmentation_mask/obj_
 - robot0_agentview_right_segmentation_mask/obj_
 
# Old paths, not used anymore
# Feature Images
target_features_subfolder: 
 - target_features/left
 - target_features/right
tool_features_subfolder: 
 - tool_features/left
 - tool_features/right

# Since we only extract the dino features for areas around the target and tool we need to store the offsets to the upper left corner of the image
# Offsets
target_offsets_subfolder: 
 - target_offsets/left
 - target_offsets/right
tool_offsets_subfolder: 
 - tool_offsets/left
 - tool_offsets/right
  
# Pointclouds
target_points_subfolder: 
 - dilated_target_points/left
 - dilated_target_points/right
tool_points_subfolder:
 - dilated_cotrack_tool_points_no_skip/left
 - dilated_cotrack_tool_points_no_skip/right
 
raw_target_points_subfolder:
 - raw_target_points/left
 - raw_target_points/right
 
raw_tool_points_subfolder:
 - raw_tool_points/left
 - raw_tool_points/right

# Paths
target_paths_subfolder:
 - dilated_cotrack_target_paths_no_skip/left
 - dilated_cotrack_target_paths_no_skip/right
tool_paths_subfolder:
 - dilated_cotrack_tool_paths_no_skip/left
 - dilated_cotrack_tool_paths_no_skip/right
 
# Visualizations
mask_vis_subfolder:
 - mask_vis/left
 - mask_vis/right
features_vis_subfolder: 
 - features_vis/left
 - features_vis/right
tracking_vis_subfolder:
 -  tracking_vis/left
 -  tracking_vis/right

# Intrinsics
extrinsic_keys:
 - robot0_agentview_left_extrinsics
 - robot0_agentview_right_extrinsics

intrinsic_keys:
 - robot0_agentview_left_intrinsics
 - robot0_agentview_right_intrinsics
 
# States Subfolder
states_subfolder: states

# Trajectories and Points
# The final folders that contain all demonstration information directly used by the pytorch datasets
prepared_data_subfolder: prepared_data_no_skip
simple_pc_prepared_data_subfolder: filtered_simple_pc_prepared_data

# fps subfolder
# Not sure if this is still used, was used for precomputing the Farthest Point Sampling (FPS) for the points since it is quite expensive
target_points_fps_subfolder: target_points_fps
tool_points_fps_subfolder: cotrack_tool_points_fps
target_paths_fps_subfolder: cotrack_target_paths_fps
tool_paths_fps_subfolder: cotrack_tool_paths_fps
fully_valid_tool_paths_fps_subfolder: cotrack_fv_tool_paths_fps

# Captions used for instance by the Owlv2+Samv2 segmenter
target_caption: "cup."
tool_caption: "Cutlery or screwdriver or spatula or long thin object or scissors."

# Robocasa orig folder
# Not sure if currently used
robocasa_prepro_folder: "/media/nic/LargeSandwich/kMPD/RoboCasa/preprocessed/PickPlaceCounterToStove/"


# CoTracking Settings:
track_length: 16
patch_size: 14
frame_dt: 1
use_all_frame_skip_offsets: True
fps_num_points: 50
erosion_iterations: 3
erosion_kernel_size: 3
```