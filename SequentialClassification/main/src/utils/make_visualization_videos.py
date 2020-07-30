import os
import glob
import argparse
import sys

from make_mediapipe_video import make_mediapipe_video
from make_features_table import make_features_table
from json_data import load_json

def make_visualization_videos(input_frames_directory, input_mediapipe_directory, users, output_project_directory, feature_type, table_video, visualization_types, frame_rate):

	features_config = load_json(os.path.join(output_project_directory, 'configs/features.json'))
	features = features_config[feature_type]

	if len(users) == 0:
		features_filepaths = glob.glob(os.path.join(input_mediapipe_directory, '**/*.data'), recursive = True)
	else:
		features_filepaths = []
		for user in users:
			features_filepaths.extend(glob.glob(os.path.join(input_mediapipe_directory, '*{}*'.format(user), '**/*.data'), recursive = True))

	print(features_filepaths)

	if table_video:
		print("Making feature table for each trial")
		make_features_table(input_mediapipe_directory, users, output_project_directory, 'trials')

	 
	for features_filepath in features_filepaths:
		filename = features_filepath.split('/')[-1]
		session, phrase, trial, _ = filename.split('.')
		
		frames_directory = os.path.join(input_frames_directory, session, phrase, trial, '*.png')
		save_directory = os.path.join(output_project_directory, 'visualization/videos', session, phrase, trial)
		table_filepath = os.path.join(output_project_directory, 'visualization/tables/trials', session, phrase, trial, '{}.{}.{}.png'.format(session, phrase, trial))

		if not os.path.exists(save_directory):
			os.makedirs(save_directory)
			make_mediapipe_video(frames_directory, features_filepath, save_directory, features, table_video, table_filepath, visualization_types, frame_rate)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_frames_directory', type = str, default = '/mnt/ExtremeSSD/ProcessingPipeline/DATA/Frames')
	parser.add_argument('--input_mediapipe_directory', type = str, default = '/mnt/ExtremeSSD/ProcessingPipeline/DATA/Mediapipe_Data_July_2020')
	parser.add_argument('--users', default = ['7-27-20_prerna_test'])
	parser.add_argument('--output_project_directory', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/July2020Mediapipe')
	parser.add_argument('--feature_type', type = str, default = 'visualization_features')
	parser.add_argument('--table_video', action = 'store_true')
	parser.add_argument('--visualization_types', default = [['mediapipe']])
	parser.add_argument('--frame_rate', type = int, default = 5)
	args = parser.parse_args()

	make_visualization_videos(args.input_frames_directory, args.input_mediapipe_directory, args.users, args.output_project_directory, args.feature_type, args.table_video, args.visualization_types, args.frame_rate)
