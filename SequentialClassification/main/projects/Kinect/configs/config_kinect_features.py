#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np
import pandas as pd

def feature_selection():
  features = ['elbow_left', 'wrist_left', 'hand_left', 'handtip_left', 'thumb_left', 'elbow_right', 'wrist_right', 'hand_right', 'handtip_right', 'thumb_right']
  
  to_nose_features = ['hand_left', 'handtip_left', 'thumb_left', 'hand_right', 'handtip_right', 'thumb_right']
  coordinates = ['x', 'y', 'z']

  columns = []
  for feature in features:
    joint_positions = [] #[f'{feature}_{coordinate}' for coordinate in coordinates]
    relative_positions = [] #[f'delta_{feature}_{coordinate}' for coordinate in coordinates]
    relative_squared_dist = [] #[f'delta_{feature}_squared_xyz']
    joint_orientation_positions = [] #[f'joint_orientation_{feature}_{orientation}' for orientation in ['x', 'y', 'z', 'w']] 
    
    relative_to_nose = [f'{feature}_to_nose_{coordinate}' for coordinate in coordinates] # 'nose' may change depending on specified origin_feature in dist_from_feature(), 27=NOSE default
    
    # if feature in to_nose_features: (performs worse with if-condition) 
    delta_relative_to_nose = [] #[f'delta_{feature}_to_nose_{coordinate}' for coordinate in coordinates]
    
    standardized_no_squared_positions = [] #[f'standardized_{feature}_{coordinate}' for coordinate in coordinates]
    standardized_squared_positions = [f'standardized_{feature}_squared_{coordinate}' for coordinate in coordinates]

    quantile_no_squared_positions = []#[f'quantile_{feature}_{coordinate}' for coordinate in coordinates]
    quantile_squared_positions = [f'quantile_{feature}_squared_{coordinate}' for coordinate in coordinates]

    feature_columns = joint_positions + relative_positions + relative_squared_dist + joint_orientation_positions
    feature_columns += relative_to_nose + delta_relative_to_nose + standardized_no_squared_positions + standardized_squared_positions
    feature_columns += quantile_no_squared_positions + quantile_squared_positions
    columns.extend(feature_columns)

  # angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]
  # columns.extend(angle_wrist_elbow)
  # distance_between_handtips = ['dist_between_handtips_squared_xyz', 'delta_dist_between_handtips_squared_xyz']
  # columns.extend(distance_between_handtips)

  return columns

def select_feature_kinect():
  
  cols = feature_selection()

  output_format = ''
  for col in cols:
    output_format += f'\t"{col}"'

    if col != cols[-1]:
      output_format += ',\n'

  print(output_format)


select_feature_kinect()
