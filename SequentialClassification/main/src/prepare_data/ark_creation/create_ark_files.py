"""Creates .ark files needed as intermediate step to creating .htk files

Methods
-------
_create_ark_file
create_ark_files
"""
import os
import glob

import pandas as pd

from .feature_selection import select_features


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


def create_ark_files(features_config: dict, verbose: bool = False) -> None:
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
    if not os.path.exists(ark_dir):
        os.makedirs(ark_dir)
        
    features_filepaths = glob.glob(features_config['features_dir'])
    
    for features_filepath in features_filepaths:

        if verbose:
            print(features_filepath)
        
        features_df = select_features(
            features_filepath, features_config['selected_features'])

        if features_df is not None:
            ark_filename = features_filepath.split('/')[-1].replace('data', 'ark')
            ark_filepath = os.path.join(ark_dir, ark_filename)
            title = '_'.join(features_filepath.split('/')[-1].split('.')[:-1])
            _create_ark_file(features_df, ark_filepath, title)


if __name__ == '__main__':

    features_filepath = '/home/trace/Documents/MS/2019-08/cs6999/copycat-ml/main/projects/test2/data/ark/richa2.Alligator_in_box.23.ark'
    features_config = '/home/trace/Documents/MS/2019-08/cs6999/copycat-ml/main/projects/test2/configs/features.config'
    features_to_extract = load_feature_names(features_config)
    
    features_df = select_features(features_filepath, features_to_extract)

    print(features_df.head())

    # if features_df is not None:
    #     ark_filename = features_filepath.split('/')[-1].replace('data', 'ark')#.lower()
    #     ark_filepath = os.path.join(ark_dir, ark_filename)
    #     title = '_'.join(features_filepath.split('/')[-1].split('.')[:-1])#.lower()
    #     _create_ark_file(features_df, ark_filepath, title)
    