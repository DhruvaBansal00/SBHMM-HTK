# CopyCat HTK Pipeline

A pipeline to run tracking Kinect, MediaPipe and AlphaPose experiments on sign language feature data using either HMMs or SBHMMs

## Getting Started

* Navigate to either `projects/Kinect`, `projects/Mediapipe` or `projects/AlphaPose` depending on which hand tracking the experiment is for
* Inside this directory, navigate to `configs/features.json` to select which features to use. `"all_features"` is the complete list of features that are available for the specific hand tracker. Select a subset of those features to add into `"selected_features"` to customize the list of features to run the experiment on. Note that `config_kinect_features.py` and `config_alphapose_features.py` allows a faster way to print out which features you want to use for Kinect and AlphaPose instead of manually copying and pasting each one.
* Inside this directory, navigate to `configs/features.json` `"feature_dir"` to select where the base directory of the features directory is located. More likely than not, this will not be required to change on the current system. Note that for AlphaPose, you are required to run `test.py` on the Data in the `"feature_dir"` so that it aligns with the naming convention the pipeline uses.
* `configs/` also contains other HMM hyperparameters such as `hhed#.conf` where it lists the number of mixture models and the number of states that will be used in the model. The current pipeline will iterate through these `hhed#.conf` files using the `train_iters` argument when running the experiment.

## Running the Pipeline

* Open a terminal at `projects/Kinect`, `projects/Mediapipe` or `projects/AlphaPose` and execute (this is one specific example): `python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 --hmm_insertion_penalty 0 --cross_val_method leave_one_user_out --n_splits 4 --cv_parallel --parallel_jobs 5 --prepare_data` 
* This will first prepare the HTK files for HMMs on all users data located at the base directory `"features_dir"` using user independent cross validation of 4 folds on 5 parallel processes. The specifics of some common options are given in the next section. Please look at `main/src/main.py` for the complete list of arguments that can be used with the pipeline.
* **This helps to stay organized and is very important.** After the results have completed, please commit and push all changes with the message: `exp <Kinect/Mediapipe> <###> [comments]` where `###` is the excel row number of the experiment. In the excel sheet, please note relevant information, especially the model, tracker type, and command as well as the **average** word, sentence, insertion, and deletion error that is printed to terminal after the experiment ends.

## Optional Edits

Note that the important ones are indicated with [**IMP**]

* [**IMP**] `driver.py --prepare_data`: If the list of users or the type of features have changed, then it is required to set this flag to recompile the data files and generate ARK/HTK files. There is no harm in always using this flag if you are unsure.
* [**IMP**] `driver.py --users`: Specify a list of users (keywords) to run the visualization on a refined list of `.data` files. If empty, then all users in `"features_dir"` are used. Be careful with left hand vs right hand datasets.
* [**IMP**] `driver.py --test_type`: The type of testing to perform (none, test on train, cross validation, standard). cross validation ('cross_val') is the most likely option for generalized testing.
* [**IMP**] `driver.py --cross_val_method`: The type of split for cross validation (kfold, leave one phrase out, stratified, leave one user out). For user independent use `leave_one_user_out`. 
* [**IMP**] `driver.py --n_splits`: The number of splits to perform in cross validation. For the current dataset ~5 is normal. Note that user independent sets this value internally, but a value is still required to pass in.
* [**IMP**] `driver.py --train_iters`: This is the iterations to perform on the model training. We are currently using something close to `25 50 75 100 120 140 160 180 200 220`. While you can change the values, remove items from the list, or add items to the list, it is unlikely this will deviate too much. Feel free to test with different values/size though. Note that this corresponds to the `hhed#.conf` files so make sure there is enough files for each element in this train iterations list.
* [**IMP**] `driver.py --cv_parallel`: If set this will run the experiment in parallel. Recommended to speed up experiment.
* [**IMP**] `driver.py --parallel_jobs`: If `cv_parallel` is set then this is the number of processes to use. 5 - 10 is recommended.
* [**IMP**] `driver.py --hmm_insertion_penalty`: This helps balance the percentage of insertions and deletions on the experiment. Ideally we want both of them to be around equal. A larger value will decrease the percentage of inserts performed.
* `driver.py --mean`: The initial mean value used for training. Unlikely to change.
* `driver.py --variance`: The initial variance value used for training. Unlikely to change.
* `driver.py --transition_prob`: The initial transition probability value used for training. Unlikely to change.
* `driver.py --start`: This determines where we start testing. Unlikely to change.
* `driver.py --end`: This determines where we end testing. Unlikely to change.
* `driver.py --method`: Whether to perform recognition or verification. Unlikely to change.
* `driver.py --random_state`: default seed for consistent results. Most likely do not modify.
* `driver.py --test_size`: The amount of data to use for testing (currently set at 0.1 for 10% of dataset). Unlikely to change.
* `driver.py --save_results`: This will append the results to a file to keep track of information across runs. Most likely you do not need to use this, and it has not been recently tested. 
* `driver.py --save_results_file`: The name of the file to save results for if `--save_results` is set. Unlikely to change.

### Additional SBHMM Arguments

* There are many SBHMM arguments available. It is not covered here.
* Note that recent experiments have only passed in the `beam_threshold` argument to set value 50,000 for HMMs. 

### Some Common Commands

* user-independent on all users/datafiles that contain the word `"subject1"` and `"subject2"`: `python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 --hmm_insertion_penalty 0 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 5 --prepare_data --users subject1 subject2`
* user-adaptive (stratified) on all users: `python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 --hmm_insertion_penalty 0 --cross_val_method stratified --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 10 --prepare_data` 
* user-dependent on all users: `python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 --hmm_insertion_penalty 0 --cross_val_method kfold --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 5 --prepare_data`

## Future Work
- Add SBHMM information to README
- Test file that stores results/configs for each training session
- ???

