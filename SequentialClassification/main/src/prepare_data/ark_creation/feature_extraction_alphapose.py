#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np
import pandas as pd

def feature_labels():
  features = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
  coordinates = ['x', 'y']

  columns = []
  for feature in features:
    joint_positions = [f'{feature}_{coordinate}' for coordinate in coordinates]
    relative_positions = [f'delta_{feature}_{coordinate}' for coordinate in coordinates]
    relative_squared_dist = [f'delta_{feature}_squared_xy']
    
    relative_to_nose = [f'delta_{feature}_to_nose_{coordinate}' for coordinate in coordinates]
    
    standardized_no_squared_positions = [f'standardized_{feature}_{coordinate}' for coordinate in coordinates]
    standardized_squared_positions = [f'standardized_{feature}_squared_{coordinate}' for coordinate in coordinates]

    feature_columns = joint_positions + relative_positions + relative_squared_dist
    feature_columns += relative_to_nose + standardized_no_squared_positions + standardized_squared_positions
    columns.extend(feature_columns)

  angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]
  columns.extend(angle_wrist_elbow)

  return columns

# can return in a list: absolute positions, relative positions, distance from a particular joint
def get_features(frame, feature_set):
  features = []
  joint_positions = [frame[feature_set][index] for index in range(2)]
  # if you want absolute positions uncomment
  features.extend(joint_positions)

  # replace feature_set with the index of the joint to want relative positions wrt for e.g. 0 for spine, 27 for nose
  new_origin_positions = [frame[0][index] for index in range(2)]
  relative = []
  dist = 0
  for i in range(2):
    dist = dist + (joint_positions[i]-new_origin_positions[i])*(joint_positions[i]-new_origin_positions[i])
    relative.append(joint_positions[i] - new_origin_positions[i])
  # if you want relative positions uncomment
  features.extend(relative)
  # if you want distance from relative positions uncomment
  # features.append(np.sqrt(dist))
  features.append(dist)
  return features

# returns angles of left wrist to left elbow and right wrist to right elbow respectively
def angle_wrist_elbow(frame):
  origin = [frame[0][index] for index in range(3)]
  elbow_left = [frame[7][index] for index in range(3)]
  wrist_left = [frame[9][index] for index in range(3)]
  elbow_right = [frame[8][index] for index in range(3)]
  wrist_right = [frame[10][index] for index in range(3)]
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
  origin = [frame[0][index] for index in range(2)]
  previous = [prev_frame[feature_set][index] for index in range(2)]
  current = [frame[feature_set][index] for index in range(2)]
  previous = [a_i - b_i for a_i, b_i in zip(previous, origin)]
  current = [a_i - b_i for a_i, b_i in zip(current, origin)]
  delta = [a_i - b_i for a_i, b_i in zip(current, previous)]
  return delta


def feature_extraction_alphapose(input_filepath: str, features_to_extract: list, scale: int = 10, drop_na: bool = True) -> pd.DataFrame:
  
  with open(input_filepath, 'r') as in_file:
    data = json.load(in_file)

  frames = data

  keypoints = [frame["keypoints"] for frame in frames]
  joint_positions = np.asarray([[frame[3*coord], frame[3*coord+1]] for frame in frames for coord in range(len(frame)/3)])
  new_joint_positions = []
  no_body_count = 0
  multi_body_count = 0
  frame_nums = np.asarray([int(frame["image_id"].split('.')[0]) for frame in frames])
  for a in range(frame_nums[-1]):
    if frame_nums.count(a) < 1:
      no_body_count+=1
    elif frame_nums.count(a) > 1:
      multi_body_count += 1

  all_positions = np.stack(joint_positions).astype(float)

  mean = np.mean(all_positions, axis=0)
  var = np.var(all_positions, axis=0)

  standardized_no_sq = (all_positions - mean)/var
  standardized_sq = np.square(all_positions - mean)/var
  standardized_count = 0

  all_features = []

  prev_frame = frames[0]

  for frame_number, frame in enumerate(joint_positions):

    features = []
    for index in range(17):
      features.extend(get_features(frame, index) + deltas(frame, prev_frame, index)) # Compare with previous version...
      features.extend(list(standardized_no_sq[standardized_count, index]))
      features.extend(list(standardized_sq[standardized_count, index]))
    
    features.extend(angle_wrist_elbow(frame))
    prev_frame = frame
    standardized_count += 1
    
    all_features.append(features)

  cols = feature_labels()

  df = pd.DataFrame(all_features, columns = cols)

  df = df.loc[:, df.columns.isin(features_to_extract)]
  if drop_na: df = df.dropna(axis=0)
  df = df * scale
  df = df.round(6)

  #print(f'AlphaPose DataFrame: {df}')
  return df

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
