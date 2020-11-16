"""Processes raw features extracted from MediaPipe/Kinect, and selects
the specified features for use during training of HMMs.

Methods
-------
_load_json
_calc_delta
_add_delta_col
select_features
"""
import json
import random
import argparse

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import cdist


def _load_json(json_file: str) -> dict:
    """Load JSON file TODO: Remove and use src.utils.load_json.

    Parameters
    ----------
    json_file : str
        File path of JSON to be loaded.

    Returns
    -------
    data : dict
        Data loaded from JSON.
    """
    
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
        
    return data


def _calc_delta(col: str) -> np.ndarray:
    """Calculates delta between consecutives rows of a given column.

    Parameters
    ----------
    col : str
        Column for which to calculate delta.

    Returns
    -------
    np.ndarray
        Delta of rows.
    """
    
    return np.concatenate([[0], col[1:].values - col[:-1].values])


def _add_delta_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate delta for a column and add it as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features.

    col : str
        Column for which to calculate delta.

    Returns
    -------
    return_val : pd.DataFrame
        DataFrame containing features with new delta column.
    """
    
    df['delta_{}'.format(col)] = _calc_delta(df[col])
    
    return df

def landmark_box_dist(landmark: list, hand: list) -> float:

    curr_landmark = np.reshape(landmark, (3,21))
    total_dist = 0
    for point in curr_landmark:
        hand_point = [[hand[0], hand[1]]]
        landmark_point = [[point[0], point[1]]]
        total_dist += cdist(hand_point, landmark_point)[0]
    
    return total_dist/(21)


def select_features(input_filepath: str, features_to_extract: list,
                    interpolation_method: str = 'spline', order: int = 3,
                    center_on_face: bool = False, center_on_pelvis: bool = False, is_2d: bool = True,
                    scale: int = 10, drop_na: bool = True, do_interpolate: bool = False, use_optical_flow = False) -> pd.DataFrame:
    """Processes raw features extracted from MediaPipe/Kinect, and
    selects the specified features for use during training of HMMs.

    Parameters
    ----------
    input_filepath : str
        File path of raw feature data to be processed and used for
        selection.

    features_to_extract : list
        Names of columns to be selected after processing features.

    interpolation_method : str, optional, by default 'spline'
        Interpolation method used to fill missing values.

    order : int, optional, by default 3
        Hyperparameter needed for certain interpolation methods.

    center_on_face : bool, optional, by default True
        Whether to center the features on the main face.

    is_2d : bool, optional, by default True
        Whether data is 2-dimensional.

    scale : int, optional, by default 10
        Raw features are scaled from 0-1, which can cause issues with
        very small means and variances in HTK. Used to scale up
        features.

    Returns
    -------
    df : pd.DataFrame
        Selected features
    """

    data = _load_json(input_filepath)
    if not data:
        return None
    data = {int(key): value for key, value in data.items()}

    n_frames = len(data)
    hands = np.zeros((2, n_frames, 5))
    landmarks = np.zeros((2, n_frames, 63))
    faces = np.zeros((1, n_frames, 12))
    optical_flow = np.zeros((2, n_frames, 32))
    pelvis_x, pelvis_y = 0,0

    for frame in sorted(data.keys()):

        if use_optical_flow and data[frame]['optical_flow'] is not None:

            x_val = data[frame]['optical_flow']["0"]
            y_val = data[frame]['optical_flow']["1"]

            optical_flow[0, frame, :] = np.array(x_val)/10000000.0
            optical_flow[1, frame, :] = np.array(y_val)/10000000.0
        
        if data[frame]['boxes'] is not None:

            visible_hands = np.array(sorted([data[frame]['boxes'][str(i)] for i in range(len(data[frame]['boxes']))], key= lambda x:x[0]))

            distances = {(i, j): cdist([hand[frame-1][:2]], [visible_hand[:2]]) 
                        for i, hand 
                        in enumerate(hands) 
                        for j, visible_hand 
                        in enumerate(visible_hands)}
            
            if len(visible_hands) == 1:
                if frame == 0:
                    for idx in range(len(hands)):
                        hands[idx][frame] = visible_hands[0][:5] 
                else:
                    sorted_distances, _ = sorted(distances.items(), key=lambda t: t[1])
                    prev_new_hand = sorted_distances[0][0]
                    prev_keep_hand = prev_new_hand ^ 0b1
                    new_hands = sorted([visible_hands[0][:5], hands[prev_keep_hand][frame-1]], key=lambda x: x[0])
                    hands[:,frame,:] = new_hands
            else:
                visible_hand_assigned = {n: False for n in range(len(visible_hands))}
                hand_assigned = {n: False for n in range(len(hands))}
                new_hands = []
                for grouping, _ in sorted(distances.items(), key=lambda t: t[1]):
                    hand, visible_hand = grouping
                    if not hand_assigned[hand] and not visible_hand_assigned[visible_hand]:
                        hand_assigned[hand] = True
                        visible_hand_assigned[visible_hand] = True
                        new_hands.append(visible_hands[visible_hand][:5])
                hands[:,frame,:] = sorted(new_hands, key=lambda x: x[0])

            if pelvis_x == 0 and pelvis_y == 0:
                hand_2 = hands[1,frame]
                if hand_2[0] >= 0.5:
                    pelvis_x = hand_2[0]
                    pelvis_y = hand_2[1]
        
        if data[frame]['landmarks'] is not None:
            if data[frame]['boxes'] is None:
                raise Exception('Red Alert: Our assumption that landmarks are only provided when we have boxes is incorrect')
            else:
                visible_landmarks = []
                for i in range(len(data[frame]['landmarks'])):
                    for j in range(len(data[frame]['landmarks'][str(i)])):
                        visible_landmarks += data[frame]['landmarks'][str(i)][str(j)]
                visible_landmarks = np.array(visible_landmarks).reshape(-1, 63)
                curr_hands = hands[:,frame,:]

                distances = {(i, j): landmark_box_dist(landmark, hand)
                            for i, hand in enumerate(curr_hands)
                            for j, landmark in enumerate(visible_landmarks)}
                if len(visible_landmarks) == 1:
                    if frame == 0:
                        for idx in range(len(landmarks)):
                            landmarks[idx][frame] = visible_landmarks[0]   
                    else:
                        sorted_distances, _ = sorted(distances.items(), key=lambda t: t[1])
                        prev_new_landmark = sorted_distances[0][0]
                        prev_keep_landmark = prev_new_landmark ^ 0b1
                        landmarks[prev_new_landmark,frame,:] = visible_landmarks[0]
                        landmarks[prev_keep_landmark,frame,:] = landmarks[prev_keep_landmark,frame-1,:]
                
                else:
                    visible_landmark_assigned = {n: False for n in range(len(visible_hands))}
                    curr_hand_assigned = {n: False for n in range(len(hands))}
                    for grouping, _ in sorted(distances.items(), key=lambda t: t[1]):
                        hand, visible_landmark = grouping
                        if not curr_hand_assigned[hand] and not visible_landmark_assigned[visible_landmark]:
                            curr_hand_assigned[hand] = True
                            visible_landmark_assigned[visible_landmark] = True
                            landmarks[hand, frame, :] = visible_landmarks[visible_landmark]
                    
        # if data[frame]['landmarks'] is not None:
        
        #     visible_landmarks = []
        #     for i in range(len(data[frame]['landmarks'])):
        #         for j in range(len(data[frame]['landmarks'][str(i)])):
        #             visible_landmarks += data[frame]['landmarks'][str(i)][str(j)]
        #     visible_landmarks = np.array(visible_landmarks).reshape(-1, 63)

        #     if len(visible_landmarks) == 1:
        #         landmarks[:, frame] = visible_landmarks[0]

        #     distances = {(i, j): cdist([landmark[frame-1]], [visible_landmark]) 
        #                 for i, landmark 
        #                 in enumerate(landmarks) 
        #                 for j, visible_landmark 
        #                 in enumerate(visible_landmarks)}

        #     visible_landmark_assigned = {n: False for n in range(len(visible_landmarks))}
        #     landmark_assigned = {n: False for n in range(len(landmarks))}

        #     for grouping, _ in sorted(distances.items(), key=lambda t: t[1]):
        #         landmark, visible_landmark = grouping
        #         if not landmark_assigned[landmark] and not visible_landmark_assigned[visible_landmark]:
        #             landmark_assigned[landmark] = True
        #             visible_landmark_assigned[visible_landmark] = True
        #             landmarks[landmark][frame] = visible_landmarks[visible_landmark]
                    
        if data[frame]['faces'] is not None:
            
            means = np.array(np.mean(np.ma.masked_equal(faces, 0), axis=1))
            visible_faces = []
            for i in range(len(data[frame]['faces'])):
                for j in range(len(data[frame]['faces'][str(i)])):
                    visible_faces += data[frame]['faces'][str(i)][str(j)]
            visible_faces = np.array(visible_faces).reshape(-1, 12)

            for visible_face in visible_faces:
                if len(faces) == 1 and not np.any(means):
                    faces[0, frame] = visible_face
                else:
                    if not np.any(np.all(np.abs(means - visible_face) < 0.04, axis=1)):
                        new_face = np.zeros((1, n_frames, 12))
                        new_face[0, frame] = visible_face
                        faces = np.concatenate([faces, new_face], axis=0)

            means = np.array(np.mean(np.ma.masked_equal(faces, 0), axis=1))
            distances = {(i, j): cdist([mean], [visible_face])[0][0] 
                        for i, mean 
                        in enumerate(means) 
                        for j, visible_face 
                        in enumerate(visible_faces)}

            face_assigned = {n: False for n in range(len(visible_faces))}
            mean_assigned = {n: False for n in range(len(means))}

            for grouping, _ in sorted(distances.items(), key=lambda t: t[1]):
                mean, face = grouping
                if not mean_assigned[mean] and not face_assigned[face]:
                    mean_assigned[mean] = True
                    face_assigned[face] = True
                    faces[mean][frame] = visible_faces[face]
                    
    means = np.array(np.mean(np.ma.masked_equal(faces, 0), axis=1))
    n_faces = faces.shape[0]
    main_face = means[np.argmax([len(set(np.nonzero(faces[i])[0])) for i in range(n_faces)])]

    select_hands = np.any(['hand' 
                           in feature 
                           for feature 
                           in features_to_extract])
    select_landmarks = np.any(['landmark' 
                               in feature 
                               for feature 
                               in features_to_extract])
    select_faces = np.any(['face' 
                                in feature
                                for feature
                                in features_to_extract]) 

    select_optical_flow = np.any(['optical_flow'
                                in feature
                                for feature
                                in features_to_extract])

    if select_hands and not np.any(hands):
        return None

    if select_landmarks and not np.any(landmarks):
        return None

    if select_faces and not np.any(faces):
        return None
    
    if use_optical_flow and select_optical_flow and not np.any(optical_flow):
        return None

    hands_ = ['left_hand', 'right_hand']
    coordinates = ['x', 'y', 'w', 'h', 'rot']
    hand_cols = [f'{hand}_{coordinate}' 
                for hand 
                in hands_ 
                for coordinate 
                in coordinates]

    hands_ = ['left', 'right']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y', 'z']
    landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in landmarks_ 
                    for coordinate 
                    in coordinates]

    faces_ = ['face_{}'.format(i) for i in range(6)]
    coordinates = ['x', 'y']
    face_cols = ['{}_{}'.format(face, coordinate)
                for face
                in faces_
                for coordinate
                in coordinates]
    
    coordinates = ['optical_flow_x', 'optical_flow_y']
    optical_flow_cols = ['{}_{}'.format(coordinate, i)
                for i
                in range(32)
                for coordinate
                in coordinates]

    cols = hand_cols + landmark_cols + face_cols + optical_flow_cols
    hands = np.concatenate([hands[0], hands[1]], axis=1)
    landmarks = np.concatenate([landmarks[0], landmarks[1]], axis=1)
    faces_to_display = faces[np.argmax([len(set(np.nonzero(faces[i])[0])) for i in range(n_faces)])]
    optical_flow = np.concatenate([optical_flow[0], optical_flow[1]], axis=1)
    all_features = np.concatenate([hands, landmarks, faces_to_display, optical_flow], axis=1)
    df = pd.DataFrame(all_features, columns=cols)

    df = df.replace(0, np.nan)

    if select_hands and do_interpolate:

        try:
            df[hand_cols] = df[hand_cols].interpolate(interpolation_method, order=order)
        except:
            print(input_filepath)
            return None

    if select_landmarks and do_interpolate:

        try:
            df[landmark_cols] = df[landmark_cols].interpolate(interpolation_method, order=order)
        except:
            print(input_filepath)
            return None

    if is_2d:
        
        z_landmark_cols = [column for column in landmark_cols if 'z' in column]
        df = df.drop(z_landmark_cols, axis=1)

    if center_on_face:
        
        x_cols = [column for column in df.columns if 'x' in column]
        y_cols = [column for column in df.columns if 'y' in column]
        
        df[x_cols] -= main_face[-2]
        df[y_cols] -= main_face[-1]

    if center_on_pelvis:
        
        x_cols = [column for column in df.columns if 'x' in column]
        y_cols = [column for column in df.columns if 'y' in column]
        
        df[x_cols] -= pelvis_x
        df[y_cols] -= pelvis_y

    df['horizontal_hand_dist'] = df['right_hand_x'] - df['left_hand_x']
    df['vertical_hand_dist'] = df['right_hand_y'] - df['left_hand_y']

    for col in df.columns:

        df = _add_delta_col(df, col)

    df = df.loc[:, df.columns.isin(features_to_extract)]
    if drop_na:
        df = df.dropna(axis=0)
    df = df * scale
    df = df.round(6)

    return df
