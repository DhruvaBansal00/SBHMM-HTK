#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np

# can return in a list: absolute positions, relative positions, distance from a particular joint, quaternions
def get_features(frame, feature_set):
  features = []
  joint_positions = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  # if you want absolute positions uncomment
  # features.extend(joint_positions)

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
  # joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
  # features.extend(joint_orientations)

  return features

# returns angles of right wrist to right elbow and left wrist to left elbow
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

def deltas(frame, prev_frame, feature_set):
  origin = [frame["bodies"][0]["joint_positions"][27][index] for index in range(3)]
  previous = [prev_frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  current = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
  previous = [a_i - b_i for a_i, b_i in zip(previous, origin)]
  current = [a_i - b_i for a_i, b_i in zip(current, origin)]
  delta = [a_i - b_i for a_i, b_i in zip(current, previous)]
  return delta

# gets absolute xyz and quaternions. This is what the kinect gives us.
# def get_coords(frame, feature_set):
#     joint_positions = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
#     joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
#     features = joint_positions + joint_orientations
#     # print(frame_number)
#     return features

def feature_extraction_kinect(input_filepath: str, features_to_extract: list, ark_filepath: str):
  
  ark_filename = ark_filepath.split('/')[-1]
  title = ark_filename.replace('.ark', "")
  feature_to_index_dict = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}

  with open(input_filepath, 'r') as in_file:
    data = json.load(in_file)

  frames = data["frames"]

  with open(ark_filepath, 'w') as out_file:
    
    out_file.write('{} [ '.format(title))

    prev_frame = None

    for frame_number, frame in enumerate(frames):

      if frame_number == 0: prev_frame = frame

      try:
        body = frame["bodies"][0]["joint_positions"]
      except IndexError:
        print("did not detect a body in frame " + str(frame_number))
      else:
        for feature in features_to_extract:
            ind = feature_to_index_dict[feature]
            features = get_features(frame, ind) + angle_wrist_elbow(frame) + deltas(frame, prev_frame, 8) + deltas(frame, prev_frame, 9) + deltas(frame, prev_frame, 10) + deltas(frame, prev_frame, 15) + deltas(frame, prev_frame, 16) + deltas(frame, prev_frame, 17)
            to_write = str(features)[1:-1].replace(',', '')

            if feature == features_to_extract[-1]: to_write += '\n'
            else: to_write += ' '

            out_file.write(to_write)

        prev_frame = frame
        
    out_file.write(']')
    out_file.close()
  

  # To convert any file individually. Otherwise just use to_ark.sh 
  # print("This file converts raw data from Kinect .json to .ark")
  # print("Usage: python feature_extraction_kinect.py input_filepath output_filepath feature_indices")
  # print("Please input the feature set that you want to generated and seperated the index by comma: \n" +\
  #   "0:     PELVIS\n" +\
  #   "1:     SPINE_NAVEL\n" +\
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
