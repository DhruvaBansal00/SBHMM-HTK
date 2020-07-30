"""Prepares training data. Creates .ark files, .htk files, wordList,
dict, grammar, and all_labels.mlf.

Methods
-------
prepare_data
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

from . import create_ark_files, create_htk_files
from .generate_text_files import generate_text_files


def prepare_data(features_config: dict, device: int = 0, users = []) -> None:
    """Prepares training data. Creates .ark files, .htk files, wordList,
    dict, grammar, and all_labels.mlf.

    Parameters
    ----------
    features_config : dict
        A dictionary defining which features to use when creating the 
        data files.
    """
    if((device!=0) and (device!=1)):
        print("Device number is 0 for MediaPipe, 1 for Kinect. Please enter correct device number.")
    else: 
        if(device==0):
            print('Creating .ark files...')
            create_ark_files(features_config, users, verbose=False, is_select_features=True)
            print('.ark files created')
        else:
            print('.ark files have already been created')
        
        print('Creating .htk files')
        create_htk_files()
        print('.htk files created')

        print('Creating .txt files')
        generate_text_files(features_config['features_dir'], device)
        print('.txt files created')
