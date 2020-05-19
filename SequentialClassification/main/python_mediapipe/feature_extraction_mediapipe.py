#!/usr/bin/env python
# coding: utf-8

import os
import glob
import sys
import argparse
import numpy as np
from scipy.interpolate import CubicSpline


def get_coords(arr, coords):
    
    return [arr[i] for i in coords]
    
    
def hand_pos(**kwargs):
    
    if kwargs['three_dim']:
        return get_coords(kwargs['arr'], [0, 1, 2, 6, 7, 8])
    else:
        hand_coords = get_coords(kwargs['arr'], [0, 1, 5, 6])
        left_hand = hand_coords[0:2] if float(hand_coords[0] <= hand_coords[2]) else hand_coords[2:4]
        right_hand = hand_coords[2:4] if float(hand_coords[0] <= hand_coords[2]) else hand_coords[0:2]
        top_hand = hand_coords[0:2] if float(hand_coords[1] <= hand_coords[3]) else hand_coords[2:4]
        bottom_hand = hand_coords[2:4] if float(hand_coords[1] <= hand_coords[3]) else hand_coords[0:2]
        return np.concatenate((left_hand, right_hand, top_hand, bottom_hand))

def hand_delta(**kwargs):
    
    return [str(val) for val in np.array(kwargs['deltas'][hand_pos][-1]) - np.array(kwargs['deltas'][hand_pos][-2])]


def hand_rot(**kwargs):
    
    if kwargs['three_dim']:
        return get_coords(kwargs['arr'], [5, 11])
    else:
        return get_coords(kwargs['arr'], [4, 9])


def hand_rot_delta(**kwargs):

    return [str(val) for val in np.array(kwargs['deltas'][hand_rot][-1]) - np.array(kwargs['deltas'][hand_rot][-2])]


def box_dims(**kwargs):

    if kwargs['three_dim']:
        return get_coords(kwargs['arr'], [3, 4, 9, 10])
    else:
        return get_coords(kwargs['arr'], [2, 3, 7, 8])


def box_dims_delta(**kwargs):

    return [str(val) for val in np.array(kwargs['deltas'][box_dims][-1]) - np.array(kwargs['deltas'][box_dims][-2])]


def hand_dist(**kwargs):

    if kwargs['three_dim']:
        return [str(float(l_i) - float(r_i)) for r_i, l_i in zip(get_coords(kwargs['arr'], [0, 1, 2]), get_coords(kwargs['arr'], [6, 7, 8]))]
    else:
        return [str(float(l_i) - float(r_i)) for r_i, l_i in zip(get_coords(kwargs['arr'], [0, 1]), get_coords(kwargs['arr'], [5, 6]))]

def hand_dist_delta(**kwargs):
        
    return [str(val) for val in np.array(kwargs['deltas'][hand_dist][-1]) - np.array(kwargs['deltas'][hand_dist][-2])]

def construct_feature_interpolation(raw_data, bounding_box_features=10):
    timestep = 0
    x = []
    y = []
    for line in raw_data:
        all_features = np.array(line.strip('\n').split(',')[:-1]).astype(np.float)
        if len(all_features) > bounding_box_features:
            x.append(timestep)
            y.append(all_features[bounding_box_features:])
        timestep += 1
    return CubicSpline(x, y)

def landmark(**kwargs):
    return kwargs['feature_interpolator'](kwargs['timestep'])

def landmark_delta(**kwargs):
    return [str(val) for val in np.array(kwargs['deltas'][landmark][-1]) - np.array(kwargs['deltas'][landmark][-2])]



if __name__ == '__main__':

    #print(sys.argv)
    #Note that this script assumes that the data files have the following format - bounding_box_features landmarks. Face data has not been incorporated yet. 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, required=True, help='Filepath for MediaPipe features file to be converted to .ark')
    parser.add_argument('--output_filepath', type=str, required=True, help='Filepath to which to save .ark file')
    parser.add_argument('--feature_indexes', type=str, default='0,1', help='Comma seperated indices referencing which features to include for .ark file')
    parser.add_argument('--three_dim', action='store_true', help='True if data is 3-dimensional')
    args = parser.parse_args()

    ind_2_func = {0: hand_pos, 1: hand_delta, 2: hand_rot, 3: hand_rot_delta, 4: box_dims, 5: box_dims_delta, 6: hand_dist, 7: hand_dist_delta, 8: landmark, 9: landmark_delta}
    delta_2_func = {hand_delta: hand_pos, hand_rot_delta: hand_rot, box_dims_delta: box_dims, hand_dist_delta: hand_dist, landmark_delta: landmark}
    if args.three_dim:
        deltas = {hand_pos: [[0, 0, 0, 0, 0, 0]], hand_rot: [[0, 0]], box_dims: [0, 0, 0, 0], hand_dist: [[0, 0, 0]]}
    else:
        deltas = {hand_pos: [[0, 0, 0, 0, 0, 0, 0, 0]], hand_rot: [[0, 0]], box_dims: [0, 0, 0, 0], hand_dist: [[0, 0]], landmark: [[0 for i in range(126)]]}

    with open(args.input_filepath, 'r') as in_file:
        data = in_file.readlines()

    # bad_files = []
    # if len(data) == 0:
    #     print('--------------------------------')
    #     print('Bad File')
    #     print(args)
    #     print(args.input_filepath)
    #     print('--------------------------------')

    if len(data):
        feature_interpolator = construct_feature_interpolation(data)

        indexes = [int(v) for v in args.feature_indexes.split(',')]
        with open(args.output_filepath, 'w') as out_file:
            
            out_file.write('_'.join(args.input_filepath.split('/')[-1].split('.')[:-1]) + ' [ ')

            timestep = 0
            for line in data:
                
                all_features = line.strip('\n').split(',')[:-1]
                for ind in indexes[:-1]:
                    if ind_2_func[ind] in delta_2_func:
                        features = delta_2_func[ind_2_func[ind]](arr=all_features, three_dim=args.three_dim, deltas=deltas, feature_interpolator=feature_interpolator, timestep=timestep)
                        deltas[delta_2_func[ind_2_func[ind]]].append([float(feat) for feat in features])
                    features = ind_2_func[ind](arr=all_features, three_dim=args.three_dim, deltas=deltas, feature_interpolator=feature_interpolator, timestep=timestep)
                    features = [str(val) for val in [round(float(feat), 5) for feat in features]]
                    out_file.write(' '.join(features) + ' ')
                if ind_2_func[indexes[-1]] in delta_2_func:
                    features = delta_2_func[ind_2_func[indexes[-1]]](arr=all_features, three_dim=args.three_dim, deltas=deltas, feature_interpolator=feature_interpolator, timestep=timestep)
                    deltas[delta_2_func[ind_2_func[indexes[-1]]]].append([float(feat) for feat in features])
                features = ind_2_func[indexes[-1]](arr=all_features, three_dim=args.three_dim, deltas=deltas, feature_interpolator=feature_interpolator, timestep=timestep)
                features = [str(val) for val in [round(float(feat), 5) for feat in features]]
                out_file.write(' '.join(features) + '\n')
                timestep += 1
                
            out_file.write(']')
            out_file.close()
            print("File " + args.output_filepath + " created")

    # if (len(sys.argv) != 5):
    #     print("This file converts raw data from Mediapipe (.data) to .ark")
    #     print("Usage: feature_extraction_mediapipe.py input_filepath output_filepath feature_indices 3Ddata")
    #     print(
    #     '''
    #     Please input the feature set that you want to generated and seperated the index by comma: \n
    #         0:     hand position\n
    #         1:     hand position delta\n
    #         2:     rotation\n
    #     '''
    #     )
    #     print("Note that 3Ddata takes either F or T (case matters)")


    # in_filepath = sys.argv[1]
    # out_filepath = sys.argv[2]
    # indexes = [int(x) for x in sys.argv[3].split(',')]
    # three_dim = sys.argv[4] == "T"

    # ind_2_func = {0: hand_pos, 1: hand_delta, 2: hand_rot, 3: hand_rot_delta, 4: box_dims, 5: box_dims_delta}

    # with open(in_filepath, 'r') as in_file:
    #     data = in_file.readlines()

    # with open(out_filepath, 'w') as out_file:
    #     out_file.write('_'.join(in_filepath.split('/')[-1].split('.')[:-1]) + ' [ ')
    #     for line in data:
            
    #         features = line.strip('\n').split(',')[:-1]
    #         for ind in indexes[:-1]:
    #             out_file.write(ind_2_func[ind](features, three_dim) + ' ')
    #         out_file.write(ind_2_func[indexes[-1]](features, three_dim) + '\n')
            
    #     out_file.write(']')
    #     out_file.close()
    #     print("File " + out_filepath + " created")

