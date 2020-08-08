import os
import argparse
import sys

from feature_selection import select_features

def mediapipe_feature_data(features_filepath, features, drop_na: bool = True):

    """Processes raw features extracted from MediaPipe/Kinect, and
    selects the specified features for visualization.

    Parameters
    ----------
    features_filepath : str
        File path of raw mediapipe feature data to be processed

    features : list of str
        The features to extract

    Returns
    -------
    df : pd.DataFrame
        Selected features from mediapipe
    """

    features_no_interpolate_df = select_features(features_filepath, features,
                                  center_on_face=False, scale=1, drop_na = drop_na, do_interpolate = False)

    print("Select Features (No Interpolation) DataFrame: ")
    print(features_no_interpolate_df)
    
    return features_no_interpolate_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_filepath', default = '/home/thad/Desktop/AndroidCaptureApp/mp_feats_20-03-25_prerna/alligator_in_box/1582398952685/Prerna.alligator_in_box.1582398952685.data')
    parser.add_argument('--features', default=[])
    args = parser.parse_args()

    mediapipe_feature_data(args.features_filepath, args.features)