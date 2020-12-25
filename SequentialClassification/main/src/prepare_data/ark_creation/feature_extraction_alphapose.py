#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np
import pandas as pd

def feature_labels():
  body_keypoints = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'Head', 'Neck', 'Hip', 'LBigToe', 'RBigToe', 'LSmallToe', 'RSmallToe', 'LHeel', 'RHeel']
  face_keypoints = [f'face_{i}' for i in range(68)]
  left_hand_keypoints = [f'left_hand_{i}' for i in range(21)]
  right_hand_keypoints = [f'right_hand_{i}' for i in range(21)]
  features = body_keypoints + face_keypoints + left_hand_keypoints + right_hand_keypoints

  coordinates = ['x', 'y']

  #nose_x, nose_y, nose_z, delta_nose_x, delta_nose_y, delta_nose_z, delta_feature_to_nose_

  columns = []
  for feature in features:
    # get_features() -> 2 + 2 + 1
    joint_positions = [f'{feature}_{coordinate}' for coordinate in coordinates] # [absolute position feature x, absolute position feature y] 
    relative_positions = [f'dist_{feature}_to_nose_{coordinate}' for coordinate in coordinates] # [absolute position feature x - absolute position of nose x, absolute position feature y - absolute position of nose y]
    relative_squared_dist = [f'dist_{feature}_to_nose_squared_xy'] # [(absolute position feature x - absolute position of nose x)^2 + (absolute position feature y - absolute position of nose y)^2]
    
    # deltas() -> 2
    delta = [f'delta_{feature}_{coordinate}' for coordinate in coordinates] # [current feature - previous feature]
    
    # standarized
    standardized_no_squared_positions = [f'standardized_{feature}_{coordinate}' for coordinate in coordinates]
    standardized_squared_positions = [f'standardized_{feature}_squared_{coordinate}' for coordinate in coordinates]

    feature_columns = joint_positions + relative_positions + relative_squared_dist
    feature_columns += delta + standardized_no_squared_positions + standardized_squared_positions
    columns.extend(feature_columns)

  angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]
  columns.extend(angle_wrist_elbow)

  return columns

# can return in a list: absolute positions, relative positions, distance from a particular joint
def get_features(frame, feature_set):
  features = []
  joint_positions = frame[feature_set]
  # if you want absolute positions uncomment
  features.extend(joint_positions)

  # replace feature_set with the index of the joint to want relative positions wrt for e.g, 0 for nose
  new_origin_positions = frame[0]
  relative = []
  squared_dist = 0
  for i in range(2):
    squared_dist += (joint_positions[i]-new_origin_positions[i]) ** 2
    relative.append(joint_positions[i] - new_origin_positions[i])
  # if you want relative positions uncomment
  features.extend(relative)
  # if you want distance from relative positions uncomment
  features.append(squared_dist)
  return features

# returns angles of left wrist to left elbow and right wrist to right elbow respectively
def angle_wrist_elbow(frame):
  origin = frame[0]
  elbow_left = frame[7]
  wrist_left = frame[9]
  elbow_right = frame[8]
  wrist_right = frame[10]
  elbow_left = [a_i - b_i for a_i, b_i in zip(elbow_left, origin)]
  wrist_left = [a_i - b_i for a_i, b_i in zip(wrist_left, origin)]
  elbow_right = [a_i - b_i for a_i, b_i in zip(elbow_right, origin)]
  wrist_right = [a_i - b_i for a_i, b_i in zip(wrist_right, origin)]
  elbow_left = np.asarray(elbow_left)
  wrist_left = np.asarray(wrist_left)
  elbow_right = np.asarray(elbow_right)
  wrist_right = np.asarray(wrist_right)
  angle1 = np.arccos(np.dot(elbow_left, wrist_left) / (np.linalg.norm(elbow_left)*np.linalg.norm(wrist_left)))
  angle2 = np.arccos(np.dot(elbow_right, wrist_right) / (np.linalg.norm(elbow_right)*np.linalg.norm(wrist_right)))
  features = [angle1, angle2]
  return features


def deltas(frame, prev_frame, feature_set):
  previous = prev_frame[feature_set]
  current = frame[feature_set]
  features = [a_i - b_i for a_i, b_i in zip(current, previous)] # delta
  return features


def feature_extraction_alphapose(input_filepath: str, features_to_extract: list, scale: int = 10, drop_na: bool = True) -> pd.DataFrame:

  with open(input_filepath, 'r') as in_file:
    data = json.load(in_file)

  frames = data

  keypoints = [frame["keypoints"] for frame in frames]
  # print(np.asarray(keypoints).shape)

  #51 points -> 17

  # 26 body keypoints
  # body_keypoints = keypoints[:, 0:26]

  # #face 68 keypoints
  # face_keypoints = keypoints[:, 26:94]

  # #left hand 21 keypoints
  # lefthand_keypoints = keypoints[:, 94:115]

  # #right hand 21 keypoints
  # righthand_keypoints = keypoints[:, 115:136]

  joint_positions = [[[kp[3*coord], kp[3*coord+1]] for coord in range(len(kp) // 3)] for kp in keypoints] # frames x 135 x 2

  no_body_count = 0
  multi_body_count = 0
  frame_nums = np.asarray([int(frame["image_id"].split('.')[0]) for frame in frames])

  if len(frame_nums) == 0:
    return
  
  for a in range(frame_nums[-1]):
    if np.count_nonzero(frame_nums == a) < 1:
      no_body_count+=1
    elif np.count_nonzero(frame_nums == a) > 1:
      multi_body_count += 1

  all_positions = np.stack(joint_positions).astype(float) # frames x 135 x 2
  mean = np.mean(all_positions, axis=0) #135 x 2
  var = np.var(all_positions, axis=0) #135 x 2

  standardized_no_sq = (all_positions - mean)/var
  standardized_sq = np.square(all_positions - mean)/var

  all_features = []

  prev_frame = joint_positions[0] #135 x 2

  for frame_number, frame in enumerate(joint_positions): # number of frames
    features = []
    for index in range(len(frame)): # number of features (135)
      features.extend(get_features(frame, index) + deltas(frame, prev_frame, index)) # Compare with previous version...
      features.extend(list(standardized_no_sq[frame_number, index]))
      features.extend(list(standardized_sq[frame_number, index]))
    
    features.extend(angle_wrist_elbow(frame))
    prev_frame = frame
    
    all_features.append(features)

  # print(np.asarray(all_features).shape)

  cols = feature_labels()
  #print(cols)
  #print(np.asarray(cols).shape)

  df = pd.DataFrame(all_features, columns = cols)

  df = df.loc[:, df.columns.isin(features_to_extract)]
  if drop_na: df = df.dropna(axis=0)
  df = df * scale
  df = df.round(6)

  #print(f'AlphaPose DataFrame: {df}')
  return df

# feature_extraction_alphapose("/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/AlphaPoseJson/Ishan_NewModels/alligator_above_bed/0000000000/alphapose_Ishan_NewModels.alligator_above_bed.0000000000.json", ['REar_x', 'Nose_x'])


  # To convert any file individually. Otherwise just use to_ark.sh 
  # print("This file converts raw data from AlphaPose .json to .ark")
  # print("Usage: python feature_extraction_alphapose.py input_filepath output_filepath feature_indices")
  # print("Please input the feature set that you want to generated and seperated the index by comma: \n" +\
# // Result for COCO (17 body parts)
#     {0,  "Nose"},
#     {1,  "LEye"},
#     {2,  "REye"},
#     {3,  "LEar"},
#     {4,  "REar"},
#     {5,  "LShoulder"},
#     {6,  "RShoulder"},
#     {7,  "LElbow"},
#     {8,  "RElbow"},
#     {9,  "LWrist"},
#     {10, "RWrist"},
#     {11, "LHip"},
#     {12, "RHip"},
#     {13, "LKnee"},
#     {14, "Rknee"},
#     {15, "LAnkle"},
#     {16, "RAnkle"},
# // Result for MPII (16 body parts)
#     {0,  "RAnkle"},
#     {1,  "Rknee"},
#     {2,  "RHip"},
#     {3,  "LHip"},
#     {4,  "LKnee"},
#     {5,  "LAnkle"},
#     {6,  "Pelv"},
#     {7,  "Thrx"},
#     {8,  "Neck"},
#     {9,  "Head"},
#     {10, "RWrist"},
#     {11, "RElbow"},
#     {12, "RShoulder"},
#     {13, "LShoulder"},
#     {14, "LElbow"},
#     {15, "LWrist"},
