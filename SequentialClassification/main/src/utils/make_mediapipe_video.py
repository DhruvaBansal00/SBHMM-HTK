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

def make_mediapipe_video(frames_directory, features_filepath, save_directory, features, table_video, table_filepath, visualization_types, frame_rate):


    """Generates visualization video(s) for a specific recording (trial) for the following types of dataframes: mediapipe, interpolate, and kalman.

    Parameters
    ----------
    frames_directory : str
        Directory path of raw images for the specific trial that were recorded from a capture device

    features_filepath : str
        File path to raw mediapipe data for the specific trial

    save_directory : str
        Directory path where the video(s) will be saved 

    features : list of str
       The names of the features to display in the visualization video(s)

    table_video : bool
        Whether or not to output video(s) for the specific trial with a table of feature information on the right-hand side

    visualization_types : 2D list of str
        Each element is a list of strings that describes the type of dataframes to include in the specific video. Each element will generate a seperate video
        options: 'mediapipe', 'interpolate', 'kalman'
        -- ex. [['mediapipe'], ['interpolate'], ['kalman'], ['mediapipe', 'interpolate', 'kalman']]
            Will generate four videos for the specific trial where the first three will be seperate videos with each one type of dataframe, and the last one will be an aggregate of all 3 dataframes

    frame_rate : int
        The frame rate of the video(s) generated

    Returns
    -------
    Generates visualization video(s) for a specific input video 

    """
    print("Making Visualization Video for {}".format(save_directory))

    mediapipe_feature_df = mediapipe_feature_data(features_filepath, features, drop_na = False)
    interpolate_feature_df = interpolate_feature_data(features_filepath, features, center_on_face = False, scale = 1, drop_na = False)
    kalman_feature_df = kalman_feature_data(features_filepath, features, drop_na = False)
    
    # A dictionary with the three different types of feature data DataFrame 
    feature_df_dict = {'mediapipe': mediapipe_feature_df, 'interpolate': interpolate_feature_df, 'kalman': kalman_feature_df}

    # A dictionary that has its key as the root word of each type of feature: {left_hand: [left_hand_x, left_hand_y, left_hand_w, left_hand_h], right_hand: [right_hand_x, ...] ...}
    features_to_extract_dict = {}
    for feature in features:
        if 'rot' in feature or 'dist' in feature or 'delta' in feature or 'top' in feature or 'bot' in feature: continue

        feature_key = '_'.join(feature.split("_")[0:-1])
        
        try: features_to_extract_dict[feature_key].append(feature)
        except KeyError: features_to_extract_dict[feature_key] = [feature]

    # List of images for the specific recording
    frames_filepaths = sorted(glob.glob(frames_directory))

    for visualization_type in visualization_types:
        draw_features(visualization_type, frames_filepaths, features_to_extract_dict, feature_df_dict, save_directory, table_video, table_filepath)
        save_video(visualization_type, save_directory, table_filepath.split('/')[-1], frame_rate)
        delete_images(len(frames_filepaths))


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

def save_video(visualization_type, save_directory, table_filename, frame_rate = 5):
    os.chdir(save_directory)

    visual_types = '_'.join(visualization_type)
    session, phrase, trial = table_filename.split('.')[0:3]

    video_filename = '.'.join((session, phrase, trial, visual_types, 'mp4'))

    os.system('ffmpeg -r {} -f image2 -s 1024x768 -i frame_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(frame_rate, video_filename))

def draw_features(visualization_type, frames_filepaths, features_to_extract_dict, feature_df_dict, save_directory, table_video, table_filepath):

    height, width, _ = cv2.imread(frames_filepaths[0]).shape

    for i, frame_filepath in enumerate(frames_filepaths):
        filename = f'frame_{i:03d}.png'
        image = cv2.imread(frame_filepath)
        height, width, _ = image.shape

        for feature in features_to_extract_dict.items():

            feature_key = feature[0]
            feature_key_to_extract = feature[1]
            color = (0, 0, 0)
        
            for df_type in visualization_type:
                if 'right' in feature_key:
                    color = (255, 0, 0) # blue
                    if 'interpolate' in df_type:
                        color = (0, 255, 0) # green
                    if 'kalman' in df_type:
                        color = (0, 255, 255) # yellow
                if 'left' in feature_key:
                    color = (0, 0, 255) # red
                    if 'interpolate' in df_type:
                        color = (153, 0, 153) # purple
                    if 'kalman' in df_type:
                        color = (0, 128, 255) # orange
                elif 'face' in feature_key:
                    color = (255, 255, 0) # light blue
                    if 'interpolate' in df_type:
                        color = (255, 255, 255) # white

                if 'hand' in feature_key:          
                    x, y, w, h = calculate_coordinates(feature_df_dict[str(df_type)].loc[i, feature_key_to_extract].values, height, width, 'hand')
                    if x and y and w and h:
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                elif 'landmark' in feature_key or 'face' in feature_key:
                    label = 'landmark' if 'landmark' in feature_key else 'face'
                    x, y = calculate_coordinates(feature_df_dict[str(df_type)].loc[i, feature_key_to_extract].values, height, width, label)
                    if x and y:
                        cv2.circle(image, (x, y), 3, color, -1)

        if table_video:
            table_image = cv2.imread(table_filepath)
            resized_table_image = cv2.resize(table_image, (width, height)) # the resized table image
            image = cv2.hconcat([image, resized_table_image]) # concatenate the feature image with the table

        new_frame_filepath = os.path.join(save_directory, filename)
        cv2.imwrite(new_frame_filepath, image)        

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

