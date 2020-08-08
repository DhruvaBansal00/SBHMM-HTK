"""Creates a new CopyCat project. Takes a project name and a directory
of features as arguments and creates a new project. New project includes
default configurations and a main script to process data, train models,
and test models.
"""

import os
import shutil
import argparse

from src.utils import load_json, dump_json

def create_new_project(project_name: str, features_dir: str) -> None:
    """Creates a project folder that can be used for training and
    testing.

    Parameters
    ----------
    project_name : str
        Name of new project.

    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """

    config_file = 'features.json'

    project_dir = os.path.join('projects', args.project_name)
    shutil.copytree('new_project_files', project_dir)

    features_config = load_json('new_project_files/configs/' + config_file)
    features_config['features_dir'] = features_dir
    features_config_path = f'projects/{project_name}/configs/' + config_file
    dump_json(features_config_path, features_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name')
    parser.add_argument('--glob_string', type=str,
                        help='String to be passed to glob() function to obtain training data.')
    args = parser.parse_args()

    create_new_project(args.project_name, args.glob_string)
    