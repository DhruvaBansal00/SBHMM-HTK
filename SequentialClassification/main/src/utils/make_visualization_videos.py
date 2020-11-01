import os
import glob
import argparse
import sys

from make_mediapipe_video import make_mediapipe_video
from make_features_table import make_features_table
from json_data import load_json

def make_visualization_videos(input_frames_directory, input_mediapipe_directory, input_feature_config_filepath, users, output_visualization_directory, feature_type, table_video, visualization_types, frame_rate):

	"""Generates visualization video(s) that display the feature(s) for a list of users for the following types of dataframes: mediapipe, interpolate, and custom.
	Parameters
	----------
	input_frames_directory : str
		Directory path to raw images
	input_mediapipe_directory : str
		Directory path to raw mediapipe data
	input_feature_config_filepath : str
		Filepath to the config file that controls which features are displayed
	users : str
		The user(s) that will have videos generated for
	output_visualization_directory : str
		Directory path to where the video(s) will be saved 
	feature_type : str
		The type of features to display in the visualization video(s)
	table_video : bool
		Whether or not to output video(s) for the specific trial with a table of feature information on the right-hand side. Recommended is True.
	visualization_types : 2D list of str
		Each element is a list of strings that describes the type of dataframes to include in the specific video. Each element will generate a seperate video
		options: 'mediapipe', 'interpolate', 'custom'
		-- ex. [['mediapipe'], ['interpolate'], ['custom'], ['mediapipe', 'interpolate', 'custom']]
			Will generate four videos for the specific trial where the first three will be seperate videos with each one type of dataframe, and the last one will be an aggregate of all 3 dataframes
	frame_rate : int
		The frame rate of the video(s) generated
	Returns
	-------
	N/A. Generates visualization video(s) with features
	"""

	features_config = load_json(input_feature_config_filepath)
	features = features_config[feature_type]

	if not users:
		features_filepaths = glob.glob(os.path.join(input_mediapipe_directory, '**.data'), recursive = True)
	else:
		features_filepaths = []
		for user in users:
			features_filepaths.extend(glob.glob(os.path.join(input_mediapipe_directory, f'*{user}**.data'), recursive = True))

	if table_video:
		print("Making feature table for each trial")
		make_features_table(input_mediapipe_directory, users, output_visualization_directory, 'trials')

	 
	for features_filepath in features_filepaths:
		filename = os.path.split(features_filepath)[1]
		session, phrase, trial, _ = filename.split('.')
		
		frames_directory = os.path.join(input_frames_directory, session, phrase, trial, '*.png')
		save_directory = os.path.join(output_visualization_directory, 'videos', session, phrase, trial)
		table_filepath = os.path.join(output_visualization_directory, 'tables', 'trials', session, phrase, trial, f'{session}.{phrase}.{trial}.png')

		if not os.path.exists(save_directory):
			os.makedirs(save_directory)
			make_mediapipe_video(frames_directory, features_filepath, save_directory, features, table_video, table_filepath, visualization_types, frame_rate)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_frames_directory', type = str, default = os.path.join('input','DATA','Frames'), help = 'Directory path to raw images.')
	parser.add_argument('--input_mediapipe_directory', type = str, default = os.path.join('input','DATA','Mediapipe'), help = 'Directory path to raw mediapipe data.')
	parser.add_argument('--input_feature_config_filepath', type = str, default = os.path.join('input','configs','features.json'), help = 'Filepath to the config file that controls which features are displayed.')
	parser.add_argument('--users', nargs='*', default = None, help = 'The user(s) that will have videos generated for.')
	parser.add_argument('--output_visualization_directory', type = str, default = os.path.join('output', 'visualization'), help = 'Directory path to where the video(s) will be saved.')
	parser.add_argument('--feature_type', type = str, default = 'visualization_features', help = 'The type of features to display in the visualization video(s).')
	parser.add_argument('--table_video', action = 'store_true', default = True, help = 'Whether or not to output video(s) for the specific trial with a table of feature information on the right-hand side. Recommended is True.')
	parser.add_argument('--visualization_types', nargs='*', default = [['mediapipe']], help = 'Each element is a list of strings that describes the type of dataframes to include in the specific video. Each element will generate a seperate video.')
	parser.add_argument('--frame_rate', type = int, default = 5, help = 'The frame rate of the video(s) generated')
	args = parser.parse_args()

	make_visualization_videos(args.input_frames_directory, args.input_mediapipe_directory, args.input_feature_config_filepath, args.users, args.output_visualization_directory, args.feature_type, args.table_video, args.visualization_types, args.frame_rate)