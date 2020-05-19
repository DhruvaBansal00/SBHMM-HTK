# copycat-ml/main
In this folder, you can:
- Create new projects, which will copy over all necessary files, including:
    - prepare_data.py
    - train.py
    - test.py
    - configs/ (containing default config files)
- Prepare data, which will:
    - Select features based on config file
    - Create ARK files
    - Create HTK files
    - Create all_labels.mlf
    - Create grammar
    - Create wordList
    - Create dict
- Train models
- Test models

The basic commands needed for each part outlined above will be described in more detail below.

## Create new projects
From the main folder, open a terminal and type:  
`python create_new_project.py --project_name [PROJECT_NAME]`

This will create a new project with PROJECT_NAME in the projects/ folder and copy the necessary files and folders over.

## Prepare data
Once a new project has been created, `cd` to that directory by typing:  
`cd projects/[PROJECT_NAME]`   

In the directory `[PROJECT_NAME]/configs` there is a file called `features.config`. This file specifies which features to select for training the models. By default, all features are included. This is too many! These need to be filtered down. In the future, after we've identified a good set of features, we can update the `features.config` to use these. But for now, experiment! Once you've updated `features.config`, you are ready to prepare the data. 

From the terminal, type:  
`python prepare_data.py --features_dir '[FEATURES_DIR]'`  
where FEATURES_DIR is the directory containing the features from which we will select. Note that FEATURES_DIR needs to be passed as a string (using single or double quotes), and it needs to include the necessary /* /* e.g.  
`'/home/thad/Desktop/AndroidCaptureApp/mp_feats_2020-03-21_prerna/*/*/*'`.  
This will create all the necessary files for training/test.

## Train
From the project folder, in the terminal, simply type  
`python train.py`  
You can view the `train.py` file to see which variables can be passed, but the command above calls the script using the defaults that have been used throughout. For now, a single model architecture is used for all words, but in the future we may modify this such that different architectures can be used for different words. 

## Test
From the project folder, in the terminal, simply type  
`python test.py`  
You can view the `test.py` file to see which variables can be passed, but the command above calls the script using the defaults that have been used throughout. Results are stored in the `hresults/` directory, and the learned parameters for the models can be found in the `models/` directory. 

## Future Work
- Different model architectures for different words
- Tool to visualize/analyze results
- File that stores results/configs for each training session
- ??

