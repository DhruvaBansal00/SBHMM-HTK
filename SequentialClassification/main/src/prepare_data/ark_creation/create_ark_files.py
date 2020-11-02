"""Creates .ark files needed as intermediate step to creating .htk files

Methods
-------
_create_ark_file
create_ark_files
"""
import os
import glob
import shutil
import tqdm

import pandas as pd

from .feature_selection import select_features
from .interpolate_feature_data import interpolate_feature_data
from .feature_extraction_kinect import feature_extraction_kinect
from scipy import stats
import numpy as np

def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    with open(ark_filepath, 'w') as out:
        out.write('{} [ '.format(title))
        
    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')

    with open(ark_filepath, 'a') as out:
        out.write(']')


def create_ark_files(features_config: dict, users: list, verbose: bool, 
                is_select_features: bool, use_optical_flow: bool) -> None:
    """Creates .ark files needed as intermediate step to creating .htk
    files

    Parameters
    ----------
    features_config : dict
        Contains features_dir and features_to_extract

    verbose : bool, optional, by default False
        Whether to print output during process.
    """

    ark_dir = os.path.join('data', 'ark')
    first_frames = []
    
    if os.path.exists(ark_dir):
        shutil.rmtree(ark_dir)

    os.makedirs(ark_dir)

    if not users:
        features_filepaths = glob.glob(os.path.join(features_config['features_dir'], '**', '*.data'), recursive = True)
        features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '**', '*.json'), recursive = True))
    else:
        features_filepaths = []
        for user in users:
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}*'.format(user), '**', '*.data'), recursive = True))
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}*'.format(user), '**', '*.json'), recursive = True))
            print(features_config['features_dir'])
    if is_select_features:
        print("Generating ark/htk using select_features data model")
    else:
        print("Generating ark/htk using interpolate_features data model")

    for features_filepath in tqdm.tqdm(features_filepaths):

        if verbose:
            print(features_filepath)

        features_filename = features_filepath.split('/')[-1]
        features_extension = features_filename.split('.')[-1]
        features_df = None

        ark_filename = features_filename.replace(features_extension, 'ark')
        ark_filepath = os.path.join(ark_dir, ark_filename)
        title = ark_filename.replace('.ark', "")

        if features_extension == 'json':
            features_df = feature_extraction_kinect(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)
        elif is_select_features:
            features_df = select_features(features_filepath, features_config['selected_features'], center_on_face = False, is_2d = False, scale = 10, 
                                    drop_na = True, do_interpolate = True, center_on_pelvis = False, use_optical_flow=use_optical_flow)

        else:
            features_df = interpolate_feature_data(features_filepath, features_config['selected_features'], center_on_face = False, is_2d = True, scale = 10, drop_na = True)

        if features_df is not None:
            _create_ark_file(features_df, ark_filepath, title)

