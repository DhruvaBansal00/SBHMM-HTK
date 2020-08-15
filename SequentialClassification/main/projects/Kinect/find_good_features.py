import os
import glob
import sys
import json
import itertools
from itertools import combinations, chain

sys.path.insert(0, '../../src/utils')
from json_data import dump_json

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
    
    feature_columns = joint_positions + relative_positions + relative_squared_dist + joint_orientation_positions + relative_to_nose
    columns.extend(feature_columns)

  angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]
  columns.extend(angle_wrist_elbow)

  return columns

def select_feature_labels():
	features = ['clavicle_left', 'shoulder_left', 'elbow_left', 'wrist_left', 'hand_left', 'handtip_left', 'thumb_left', 'clavicle_right', 'shoulder_right', 'elbow_right', 'wrist_right', 'hand_right', 'handtip_right', 'thumb_right', 'nose']
	coordinates = ['x', 'y', 'z']
	columns = []

	joint_positions = [f'{feature}_{coordinate}' for feature in features for coordinate in coordinates]
	relative_positions = [f'delta_{feature}_{coordinate}' for feature in features for coordinate in coordinates]
	relative_squared_dist = [f'delta_{feature}_squared_xyz' for feature in features]
	joint_orientation_positions = [f'joint_orientation_{feature}_{orientation}' for feature in features for orientation in ['x', 'y', 'z', 'w']] 
	angle_wrist_elbow = [f'angle_wrist_elbow_{hand}' for hand in ['left', 'right']]

	columns = [joint_positions, relative_positions, relative_squared_dist, joint_orientation_positions, angle_wrist_elbow]

	return columns

def findsubsets(s, n): 
	return list(map(set, itertools.combinations(s, n)))

def get_features(subset, columns):
	features = []

	for idx in subset:
		features.append(columns[idx])

	features = list(chain.from_iterable(features))
	return features


all_features = feature_labels()

filtered_features = select_feature_labels()

indices = {0, 1, 2, 4} #skip 3 for now

subsets = []
for size in range(1, len(indices) + 1):
	subsets += findsubsets(indices, size)


for subset in subsets:
	print(subset)
	features = get_features(subset, filtered_features)
	features_json_dict = {'all_features': all_features, 'selected_features': features, 'features_dir': "/mnt/ExtremeSSD/ProcessingPipeline/DATA/Kinect_Data_July_2020"}
	dump_json('configs/features.json', features_json_dict)

	# call driver with commands
	os.system('python3 driver.py --prepare_data --save_results --test_type cross_val --users Matthew Linda David --train_iters 25 50 75 100 120 140 160 --hmm_insertion_penalty -80 --cross_val_method stratified --n_splits 5 --cv_parallel --parallel_jobs 5')