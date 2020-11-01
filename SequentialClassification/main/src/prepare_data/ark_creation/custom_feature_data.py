import numpy as np
import pandas as pd

def custom_feature_data(features_filepath, features, drop_na: bool = True):

    hands_ = ['left_hand', 'right_hand']
    coordinates = ['x', 'y', 'w', 'h']
    hand_cols = [f'{hand}_{coordinate}' 
                for hand 
                in hands_ 
                for coordinate 
                in coordinates]

    hands_ = ['left', 'right']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y']
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

    cols = hand_cols + landmark_cols + face_cols

    all_features = np.zeros((1, len(cols)))

    df = pd.DataFrame(all_features, columns=cols)

    if drop_na: df = df.dropna(axis=0)

    # print("Custom DataFrame: ")
    # print(df)

    return df