import os
import glob
import argparse
import sys
import math
import cv2
import pandas as pd

from json_data import load_json

sys.path.insert(0, '../prepare_data/ark_creation')
from mediapipe_feature_data import mediapipe_feature_data
from interpolate_feature_data import interpolate_feature_data
from kalman_feature_data import kalman_feature_data

def make_mediapipe_video(image_dir, features_filepath, save_dir, features, shared_directory, table_video, video_type_lists, frame_rate):


    """Generates visualization video(s) for a specific recording (trial) for the following types of dataframes: mediapipe, interpolate, and kalman.

    Parameters
    ----------
    image_dir : str
        Directory path of raw images for the specific trial that were recorded from a capture device

    features_filepath : str
        File path to raw mediapipe data for the specific trial

    save_dir : str
        Directory path where the video(s) will be saved 

    select_features_filepath : str
        File path to the features that should be selected from the raw mediapipe feature data for the visualization video(s)

    shared_directory : str (might remove later as it can be generated)
        The name of the phrase and trial number for the specific recording: "<phrase>/<trial_number>"

    table_video : bool
        Whether or not to output video(s) for the specific trial with a table of feature information on the right-hand side

    video_type_lists : 2D list of str
        Each element is a list of strings that describes the type of dataframes to include in the specific video. Each element will generate a seperate video
        options: 'mediapipe', 'interpolate', 'kalman'
        -- ex. [['mediapipe'], ['interpolate'], ['kalman'], ['mediapipe', 'interpolate', 'kalman']]
            Will generate four videos for the specific trial where the first three will be seperate videos with each one type of dataframe, and the last one will be an aggregate of all 3 dataframes

    frame_rate : int
        The frame rate of the video(s) generated

    Returns
    -------
    Generates video(s) for visualization purposes for a specific trial 

    """

    # FEATURES_TO_EXTRACT = ['left_hand_x', 'left_hand_y', 'left_hand_w', 'left_hand_h', 'right_hand_x', 'right_hand_y', 'right_hand_w', 'right_hand_h', 'left_landmark_0_x', 'left_landmark_0_y', 'left_landmark_1_x', 'left_landmark_1_y']

    mediapipe_feature_df = mediapipe_feature_data(features_filepath, features, drop_na = False)
    #interpolate_feature_df = interpolate_feature_data(features_filepath, features, drop_na = False)
    #kalman_feature_df = kalman_feature_data(features_filepath, features, drop_na = False)
    
    # A dictionary with the three different types of feature data DataFrame 
    #df_dict = {'mediapipe': mediapipe_feature_df, 'interpolate': interpolate_feature_df, 'kalman': kalman_feature_df}
    df_dict = {'mediapipe': mediapipe_feature_df}

    # List of images for the specific recording
    image_filepaths = sorted(glob.glob(os.path.join(image_dir, '*png')))

    # A dictionary that has its key as the root word of each type of feature: {left_hand: [left_hand_x, left_hand_y, left_hand_w, left_hand_h], right_hand: [right_hand_x, ...] ...}

    features_to_extract_dict = {}
    for feature in features:
        if 'rot' in feature or 'dist' in feature or 'delta' in feature or 'top' in feature or 'bot' in feature:
            continue
        feature_key = '_'.join(feature.split("_")[0:-1])
        if feature_key not in features_to_extract_dict:
            features_to_extract_dict[feature_key] = []
        features_to_extract_dict[feature_key].append(feature)

    print("creating video(s) for {}".format(shared_directory))

    for video_type in video_type_lists:
        draw_features(video_type, image_filepaths, features_to_extract_dict, df_dict, save_dir, shared_directory, table_video)
        save_video(video_type, save_dir, shared_directory, frame_rate)
        delete_images(len(image_filepaths))


def calculate_coordinates(data, height, width, feature_type):
    if feature_type == 'hand':
        x, y, w, h = data
        if math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h) or w == 0 or h == 0:
            x, y, w, h = None, None, None, None
        else:
            w = int(w * width)
            h = int(h * height)
            x = int(x * width - w / 2)
            y = int(y * height - h / 2)
        return (x, y, w, h)

    elif feature_type == 'landmark':
        x, y = data
        if math.isnan(x) or math.isnan(y) or x == 0 or y == 0:
            x, y = None, None
        else:
            x = int(x * width)
            y = int(y * height)
        return (x, y)
    
    elif feature_type == 'face':
        x, y = data
        if math.isnan(x) or math.isnan(y) or x == 0 or y == 0:
            x, y = None, None
        else:
            x = int(x * width)
            y = int(y * height)
        return (x, y)

def delete_images(num_frames):
    for i in range(num_frames):
        filename = f'frame_{i:03d}.png'
        os.remove(filename)

def save_video(video_type, save_dir, shared_directory, frame_rate = 5):
    os.chdir(save_dir)

    video_name_0 = '_'.join(video_type)
    video_name_1 = shared_directory.replace('/','_')

    os.system('ffmpeg -r {} -f image2 -s 1024x768 -i frame_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}_{}.mp4'.format(frame_rate, video_name_0, video_name_1))

def draw_features(video_type, image_filepaths, features_to_extract_dict, df_dict, save_dir, shared_directory, table_video):

    height, width, _ = cv2.imread(image_filepaths[0]).shape

    for i, image_filepath in enumerate(image_filepaths):
        filename = f'frame_{i:03d}.png'
        image = cv2.imread(image_filepath)
        #print(filename)
        height, width, _ = cv2.imread(image_filepaths[i]).shape

        for feature in features_to_extract_dict.items():

            feature_key = feature[0]
            feature_key_to_extract = feature[1]
            color = (0, 0, 0)
        
            for df_type in video_type:
                if 'right' in feature_key:
                    color = (255, 0, 0) # blue
                    if 'interpolate' in df_type:
                        color = (0, 255, 255) # yellow
                    if 'kalman' in df_type:
                        color = (0, 255, 0) # green
                if 'left' in feature_key:
                    color = (0, 0, 255) # red
                    if 'interpolate' in df_type:
                        color = (0, 128, 255) # orange
                    if 'kalman' in df_type:
                        color = (153, 0, 153) # purple
                elif 'face' in feature_key:
                    color = (255, 255, 0) # light blue
                    if 'interpolate' in df_type:
                        color = (255, 255, 255) # white

                if 'hand' in feature_key:          
                    x, y, w, h = calculate_coordinates(df_dict[str(df_type)].loc[i, feature_key_to_extract].values, height, width, 'hand')
                    if x and y and w and h:
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                elif 'landmark' in feature_key or 'face' in feature_key:
                    label = 'landmark' if 'landmark' in feature_key else 'face'
                    x, y = calculate_coordinates(df_dict[str(df_type)].loc[i, feature_key_to_extract].values, height, width, label)
                    if x and y:
                        cv2.circle(image, (x, y), 3, color, -1)

        if table_video:
            visualization_directory = '/'.join(save_dir.split('/')[:-3]) # directory path to <base_project>/visualization
            table_trial_directory = os.path.join(visualization_directory, 'tables/trials/', shared_directory) # directory path to <base_project>/visualization/tables/trials/<phrase>/<trial>
            table_filename = shared_directory.replace('/', '.')

            table_filepath = glob.glob(os.path.join(table_trial_directory, '*.png'))[0] # the path to the .png table
            table_image = cv2.imread(table_filepath)
            resized_table_image = cv2.resize(table_image, (width, height)) # the resized table image

            image = cv2.hconcat([image, resized_table_image]) # concatenate the feature image with the table

        save_image_filepath = os.path.join(save_dir, filename)
        cv2.imwrite(save_image_filepath, image)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default = '/home/thad/Desktop/AndroidCaptureApp/DATA/Prerna_04_07_20/alligator_in_box/1582398952685')
    parser.add_argument('--features_filepath', default = '/home/thad/Desktop/AndroidCaptureApp/mp_feats_20-03-25_prerna/alligator_in_box/1582398952685/Prerna.alligator_in_box.1582398952685.data')
    parser.add_argument('--save_dir', default = '/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/visualization/test_alligator_in_box/1582398952685')
    parser.add_argument('--features', default = []) ##change this
    parser.add_argument('--shared_directory', default = 'test_alligator_in_box/1582398952685')
    parser.add_argument('--table_video', action = 'store_true')
    parser.add_argument('--video_type_lists', default = [['mediapipe'], ['interpolate'], ['kalman'], ['mediapipe', 'interpolate', 'kalman']])
    parser.add_argument('--frame_rate', default = 5)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print("Making Directory ", args.save_dir)

    make_mediapipe_video(args.image_dir, args.features_filepath, args.save_dir, args.features, args.shared_directory, args.table_video, args.video_type_lists, args.frame_rate)

