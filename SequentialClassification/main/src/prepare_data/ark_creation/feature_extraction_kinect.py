#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np
import pandas as pd

def feature_labels():
  features = ['pelvis', 'spine_naval', 'spine_chest', 'neck', 'clavicle_left', 'shoulder_left', 'elbow_left', 'wrist_left', 'hand_left', 'handtip_left', 'thumb_left', 'clavicle_right', 'shoulder_right', 'elbow_right', 'wrist_right', 'hand_right', 'handtip_right', 'thumb_right', 'hip_left', 'knee_left', 'ankle_left', 'foot_left', 'hip_right', 'knee_right', 'ankle_right', 'foot_right', 'head', 'nose', 'eye_left', 'ear_left', 'eye_right', 'ear_right']
  coordinates = ['x', 'y', 'z']

  columns = []
  for feature in features:
    joint_positions = [f'{feature}_{coordinate}' for coordinate in coordinates]
    relative_positions = [f'delta_{feature}_{coordinate}' for coordinate in coordinates]
    relative_squared_dist = [f'delta_{feature}_squared_xyz']
    joint_orientation_positions = [f'joint_orientation_{feature}_{orientation}' for orientation in ['x', 'y', 'z', 'w']] 
    
    relative_to_nose = [f'delta_{feature}_to_nose_{coordinate}' for coordinate in coordinates]
    
    standardized_no_squared_positions = [f'standardized_{feature}_{coordinate}' for coordinate in coordinates]
    standardized_squared_positions = [f'standardized_{feature}_squared_{coordinate}' for coordinate in coordinates]

    feature_columns = joint_positions + relative_positions + relative_squared_dist + joint_orientation_positions
    feature_columns += relative_to_nose + standardized_no_squared_positions + standardized_squared_positions
    columns.extend(feature_columns)

  angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]
  columns.extend(angle_wrist_elbow)
  distance_between_handtips = ['dist_between_handtips_squared_xyz', 'delta_dist_between_handtips_squared_xyz']
  columns.extend(distance_between_handtips)

  return columns

# can return in a list: absolute positions, relative positions, distance from a particular joint, quaternions
def get_features(frame, feature_set):
  features = []
  joint_positions = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  # if you want absolute positions uncomment
  features.extend(joint_positions)

  # replace feature_set with the index of the joint to want relative positions wrt for e.g. 0 for spine, 27 for nose
  new_origin_positions = [frame["bodies"][0]["joint_positions"][27][index] for index in range(3)]
  relative = []
  dist = 0
  for i in range(3):
    dist = dist + (joint_positions[i]-new_origin_positions[i])*(joint_positions[i]-new_origin_positions[i])
    relative.append(joint_positions[i] - new_origin_positions[i])
  # if you want relative positions uncomment
  features.extend(relative)
  # if you want distance from relative positions uncomment
  # features.append(np.sqrt(dist))
  features.append(dist)

  # if you want quaternions uncomment
  joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
  features.extend(joint_orientations)
  #print("len get_features()", len(features))
  return features

# returns angles of left wrist to left elbow and right wrist to right elbow respectively
def angle_wrist_elbow(frame):
  origin = [frame["bodies"][0]["joint_positions"][27][index] for index in range(3)]
  elbow_left = [frame["bodies"][0]["joint_positions"][6][index] for index in range(3)]
  wrist_left = [frame["bodies"][0]["joint_positions"][7][index] for index in range(3)]
  elbow_right = [frame["bodies"][0]["joint_positions"][13][index] for index in range(3)]
  wrist_right = [frame["bodies"][0]["joint_positions"][14][index] for index in range(3)]
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

def distance_between_handtips(frame, prev_frame):
  curr_left_handtip = [frame["bodies"][0]["joint_positions"][9][index] for index in range(3)]
  curr_right_handtip = [frame["bodies"][0]["joint_positions"][16][index] for index in range(3)]
  curr_dist_between_handtips = [a_i - b_i for a_i, b_i in zip(curr_left_handtip, curr_right_handtip)]
  # print("start", curr_dist_between_handtips)
  prev_left_handtip = [prev_frame["bodies"][0]["joint_positions"][9][index] for index in range(3)]
  prev_right_handtip = [prev_frame["bodies"][0]["joint_positions"][16][index] for index in range(3)]
  prev_dist_between_handtips = [a_i - b_i for a_i, b_i in zip(prev_left_handtip, prev_right_handtip)]
  # print(prev_dist_between_handtips)
  delta_dist_between_handtips = [a_i - b_i for a_i, b_i in zip(curr_dist_between_handtips, prev_dist_between_handtips)]
  # print(delta_dist_between_handtips)
  curr_dist_between_handtips_squared_xyz = sum([a_i*a_i for a_i in curr_dist_between_handtips])
  delta_dist_between_handtips_squared_xyz = sum([a_i*a_i for a_i in delta_dist_between_handtips])
  features = [curr_dist_between_handtips_squared_xyz, delta_dist_between_handtips_squared_xyz]
  # print("end",features)
  return features

def dist_from_pelvis(frame, feature_set):
  origin = [frame["bodies"][0]["joint_positions"][0][index] for index in range(3)]
  current = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  current = [a_i - b_i for a_i, b_i in zip(current, origin)]
  return current

def deltas(frame, prev_frame, feature_set):
  origin = [frame["bodies"][0]["joint_positions"][0][index] for index in range(3)]
  previous = [prev_frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  current = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  previous = [a_i - b_i for a_i, b_i in zip(previous, origin)]
  current = [a_i - b_i for a_i, b_i in zip(current, origin)]
  delta = [a_i - b_i for a_i, b_i in zip(current, previous)]
  #print("len deltas()=>", len(delta))
  return current

# gets absolute xyz and quaternions. This is what the kinect gives us.
# def get_coords(frame, feature_set):
#     joint_positions = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
#     joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
#     features = joint_positions + joint_orientations
#     # print(frame_number)
#     return features

def feature_extraction_kinect(input_filepath: str, features_to_extract: list, scale: int = 10, drop_na: bool = True) -> pd.DataFrame:
  
  with open(input_filepath, 'r') as in_file:
    data = json.load(in_file)

  frames = data["frames"]

  joint_positions = np.asarray(data["frames"])
  new_joint_positions = []
  no_body_count = 0
  multi_body_count = 0
  for a in joint_positions:
    try:
      frame = a["bodies"][0]["joint_positions"]
      new_joint_positions.append(a)
    except IndexError:
      no_body_count += 1

    try:
      body2 = a["bodies"][1]["joint_positions"]
      multi_body_count += 1
    except IndexError:
      pass

  all_positions = np.stack([np.array(a['bodies'][0]['joint_positions']) for a in new_joint_positions])
  all_positions = all_positions.astype(float)

  mean = np.mean(all_positions, axis=0)
  var = np.var(all_positions, axis=0)

  standardized_no_sq = (all_positions - mean)/var
  standardized_sq = np.square(all_positions - mean)/var
  standardized_count = 0

  all_features = []

  prev_frame = frames[0]

  for frame_number, frame in enumerate(frames):

    features = []
    
    try:
      body = frame["bodies"][0]["joint_positions"]
    except IndexError:
      print("did not detect a body in frame " + str(frame_number))
    else:
      for index in range(32):
        features.extend(get_features(frame, index) + deltas(frame, prev_frame, index)) # Compare with previous version...
        features.extend(list(standardized_no_sq[standardized_count, index]))
        features.extend(list(standardized_sq[standardized_count, index]))
        #print("feature_ex_ki()=> ", frame_number, len(features))
        # print(features)
      
      features.extend(angle_wrist_elbow(frame))
      features.extend(distance_between_handtips(frame, prev_frame))
      #print("all", len(features))
      #print(features)
      #features = features[1:-1] # Ensure this is right (dropping empty stuff)!
      prev_frame = frame
      standardized_count += 1
    
    all_features.append(features)

  cols = feature_labels()

  df = pd.DataFrame(all_features, columns = cols)

  df = df.loc[:, df.columns.isin(features_to_extract)]
  if drop_na: df = df.dropna(axis=0)
  df = df * scale
  df = df.round(6)

  #print(f'Kinect DataFrame: {df}')
  return df

  # To convert any file individually. Otherwise just use to_ark.sh 
  # print("This file converts raw data from Kinect .json to .ark")
  # print("Usage: python feature_extraction_kinect.py input_filepath output_filepath feature_indices")
  # print("Please input the feature set that you want to generated and seperated the index by comma: \n" +\
  #   "0:     PELVIS\n" +\
  #   "1:     SPINE_NAVAL\n" +\
  #   "2:     SPINE_CHEST\n" +\
  #   "3:     NECK\n" +\
  #   "4:     CLAVICLE_LEFT\n" +\
  #   "5:     SHOULDER_LEFT\n" +\
  #   "6:     ELBOW_LEFT\n" +\
  #   "7:     WRIST_LEFT\n" +\
  #   "8:     HAND_LEFT\n" +\
  #   "9:     HANDTIP_LEFT\n" +\
  #   "10:    THUMB_LEFT\n" +\
  #   "11:    CLAVICLE_RIGHT\n" +\
  #   "12:    SHOULDER_RIGHT\n" +\
  #   "13:    ELBOW_RIGHT\n" +\
  #   "14:    WRIST_RIGHT\n" +\
  #   "15:    HAND_RIGHT\n" +\
  #   "16:    HANDTIP_RIGHT\n" +\
  #   "17:    THUMB_RIGHT\n" +\
  #   "18:    HIP_LEFT\n" +\
  #   "19:    KNEE_LEFT\n" +\
  #   "20:    ANKLE_LEFT\n" +\
  #   "21:    FOOT_LEFT\n" +\
  #   "22:    HIP_RIGHT\n" +\
  #   "23:    KNEE_RIGHT\n" +\
  #   "24:    ANKLE_RIGHT\n" +\
  #   "25:    FOOT_RIGHT\n" +\
  #   "26:    HEAD\n" +\
  #   "27:    NOSE\n" +\
  #   "28:    EYE_LEFT\n" +\
  #   "29:    EAR_LEFT\n" +\
  #   "30:    EYE_RIGHT\n" +\
  #   "31:    EAR_RIGHT\n")
