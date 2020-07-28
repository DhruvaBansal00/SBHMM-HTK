#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import json
import numpy as np

intrinsic = np.array([[900, 0, 940],[0, 900, 560], [0, 0, 1]])


# can return in a list: absolute positions, relative positions, distance from a particular joint, quaternions
def get_features(frame, feature_set, points2d):
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
    relative.append((joint_positions[i] - new_origin_positions[i]))
  # if you want relative positions uncomment
  features.extend(relative)
  # features.extend(points2d[feature_set]*100 - points2d[27]*100)
  # if you want distance from relative positions uncomment
  # features.append(np.sqrt(dist))
  features.append(dist)

  # if you want quaternions uncomment
  # joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
  # features.extend(joint_orientations)

  return features

def convert_points_2d(joint_positions, intrinsic):
  joint_positions = joint_positions.transpose()
  joint_positions = joint_positions/joint_positions[-1]
  rgb = intrinsic @ joint_positions
  rgb = rgb[0:-1, :]
  rgb = rgb.transpose()
  return rgb

def calculate_angle(one, two):
  return np.arccos(np.dot(one, two) / (np.linalg.norm(one)*np.linalg.norm(two)))

def get_pitch_roll_yaw(joint_positions):
  features = []

  right_shoulder_to_elbow = (joint_positions[12]-joint_positions[13])/np.linalg.norm(joint_positions[12]-joint_positions[13])
  right_neck_to_shoulder = (joint_positions[3]-joint_positions[12])/np.linalg.norm(joint_positions[3]-joint_positions[12])
  # features.append(np.arctan2(right_shoulder_to_elbow[0], right_neck_to_shoulder[0]) * 180 / np.pi)
  # features.append(np.arctan2(right_shoulder_to_elbow[1], right_neck_to_shoulder[1]) * 180 / np.pi)
  # features.append(np.arctan2(right_shoulder_to_elbow[2], right_neck_to_shoulder[2]) * 180 / np.pi)
  features.append(calculate_angle(right_neck_to_shoulder, right_shoulder_to_elbow))
  
  left_shoulder_to_elbow = (joint_positions[5]-joint_positions[6])/np.linalg.norm(joint_positions[5]-joint_positions[6])
  left_neck_to_shoulder = (joint_positions[3]-joint_positions[5])/np.linalg.norm(joint_positions[3]-joint_positions[5])
  # features.append(np.arctan2(left_shoulder_to_elbow[0], left_neck_to_shoulder[0]) * 180 / np.pi)
  # features.append(np.arctan2(left_shoulder_to_elbow[1], left_neck_to_shoulder[1]) * 180 / np.pi)
  # features.append(np.arctan2(left_shoulder_to_elbow[2], left_neck_to_shoulder[2]) * 180 / np.pi)
  features.append(calculate_angle(left_neck_to_shoulder, left_shoulder_to_elbow))
  
  right_shoulder_to_elbow = (joint_positions[12]-joint_positions[13])/np.linalg.norm(joint_positions[12]-joint_positions[13])
  right_elbow_to_wrist = (joint_positions[13]-joint_positions[14])/np.linalg.norm(joint_positions[13]-joint_positions[14])
  # features.append(np.arctan2(right_shoulder_to_elbow[0], right_elbow_to_wrist[0]) * 180 / np.pi)
  # features.append(np.arctan2(right_shoulder_to_elbow[1], right_elbow_to_wrist[1]) * 180 / np.pi)
  # features.append(np.arctan2(right_shoulder_to_elbow[2], right_elbow_to_wrist[2]) * 180 / np.pi)
  features.append(calculate_angle(right_shoulder_to_elbow, right_elbow_to_wrist))

  left_shoulder_to_elbow = (joint_positions[5]-joint_positions[6])/np.linalg.norm(joint_positions[5]-joint_positions[6])
  left_elbow_to_wrist = (joint_positions[6]-joint_positions[7])/np.linalg.norm(joint_positions[6]-joint_positions[7])
  # features.append(np.arctan2(left_shoulder_to_elbow[0], left_elbow_to_wrist[0]) * 180 / np.pi)
  # features.append(np.arctan2(left_shoulder_to_elbow[1], left_elbow_to_wrist[1]) * 180 / np.pi)
  # features.append(np.arctan2(left_shoulder_to_elbow[2], left_elbow_to_wrist[2]) * 180 / np.pi)
  features.append(calculate_angle(left_shoulder_to_elbow, left_elbow_to_wrist))

  return features

# can return in a list: absolute positions, relative positions, distance from a particular joint, quaternions
def get_pose_features_zahoor(joint_positions):

  # UNIT VECTORS
  features = np.array([])
  # shoulder to elbow
  features = np.append(features, (joint_positions[12]-joint_positions[13])/np.linalg.norm(joint_positions[12]-joint_positions[13]))
  features = np.append(features, (joint_positions[5]-joint_positions[6])/np.linalg.norm(joint_positions[5]-joint_positions[6]))
  # elbow to wrist
  features = np.append(features, (joint_positions[13]-joint_positions[14])/np.linalg.norm(joint_positions[13]-joint_positions[14]))
  features = np.append(features, (joint_positions[6]-joint_positions[7])/np.linalg.norm(joint_positions[6]-joint_positions[7]))
  # wrist to hand
  features = np.append(features, (joint_positions[14]-joint_positions[15])/np.linalg.norm(joint_positions[14]-joint_positions[15]))
  features = np.append(features, (joint_positions[7]-joint_positions[8])/np.linalg.norm(joint_positions[7]-joint_positions[8]))
  # hand to handtip
  features = np.append(features, (joint_positions[15]-joint_positions[16])/np.linalg.norm(joint_positions[15]-joint_positions[16]))
  features = np.append(features, (joint_positions[8]-joint_positions[9])/np.linalg.norm(joint_positions[8]-joint_positions[9]))
  # wrist to handtip
  features = np.append(features, (joint_positions[14]-joint_positions[16])/np.linalg.norm(joint_positions[14]-joint_positions[16]))
  features = np.append(features, (joint_positions[7]-joint_positions[9])/np.linalg.norm(joint_positions[7]-joint_positions[9]))
  # wrist to thumb
  features = np.append(features, (joint_positions[14]-joint_positions[17])/np.linalg.norm(joint_positions[14]-joint_positions[17]))
  features = np.append(features, (joint_positions[7]-joint_positions[10])/np.linalg.norm(joint_positions[7]-joint_positions[10]))
  # right to left shoulder
  features = np.append(features, (joint_positions[12]-joint_positions[5])/np.linalg.norm(joint_positions[12]-joint_positions[5]))
  # right handtip to left handtip
  features = np.append(features, (joint_positions[16]-joint_positions[9])/np.linalg.norm(joint_positions[16]-joint_positions[9]))
  # # right elbow to left elbow
  # features = np.append(features, (joint_positions[13]-joint_positions[6])/np.linalg.norm(joint_positions[13]-joint_positions[6]))
  
  return list(features)

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
  delta = [a_i*100 - b_i*100 for a_i, b_i in zip(current, previous)]
  return delta

# gets absolute xyz and quaternions. This is what the kinect gives us.
# def get_coords(frame, feature_set):
#     joint_positions = [frame["bodies"][0]["joint_positions"][feature_set][index] for index in range(3)]
#     joint_orientations = [frame["bodies"][0]["joint_orientations"][feature_set][index] for index in range(4)]
#     features = joint_positions + joint_orientations
#     # print(frame_number)
#     return features

if __name__ == '__main__':

  # To convert any file individually. Otherwise just use to_ark.sh 
  print("converting...")
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

  in_filepath = sys.argv[1]
  out_filepath = sys.argv[2]
  indexes = [int(x) for x in sys.argv[3].split(',')]

  with open(in_filepath, 'r') as in_file:
    data = json.load(in_file)

  frames = data["frames"]

  with open(out_filepath, 'w') as out_file:
    out_file.write('_'.join(in_filepath.split('/')[-1].split('.')[:-1]) + ' [ ')
    frame_number = 0
    prev_frame = None
    for frame in frames:
      if(frame_number==0):
        prev_frame = frame
      try:
        body = frame["bodies"][0]["joint_positions"]
        # quaternions = frame["bodies"][0]["joint_orientations"]
      except IndexError:
        print("did not detect a body in frame " + str(frame_number))
      else:
        points2d = convert_points_2d(np.array(body), intrinsic)
        points2d = points2d.astype(int)
        
        # for ind in indexes[:-1]:
        #     # features = get_features(frame, ind)
        #     features = get_features(frame, ind, points2d) 
        #     out_file.write(str(features)[1:-1].replace(',', '') + ' ') 
        # features = get_features(frame, indexes[-1], points2d) 
        # features = get_features(frame, indexes[-1], points2d) + angle_wrist_elbow(frame) + deltas(frame, prev_frame, 8) + deltas(frame, prev_frame, 9) + deltas(frame, prev_frame, 10) + deltas(frame, prev_frame, 15) + deltas(frame, prev_frame, 16) + deltas(frame, prev_frame, 17)
        
        features = get_pose_features_zahoor(np.array(body)) + angle_wrist_elbow(frame)


        out_file.write(str(features)[1:-1].replace(',', '') + '\n')
        prev_frame = frame
      frame_number = frame_number + 1
        
    out_file.write(']')
    out_file.close()