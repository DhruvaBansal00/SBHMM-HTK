import os
import glob
import argparse
import sys

from make_mediapipe_video import make_mediapipe_video
from make_features_table import make_features_table
from json_data import load_json

def make_visualization_videos(base_image_dir, base_features_dir, users, base_project_dir, feature_type, table_video, video_type_lists, frame_rate):

	image_dirs = []
	for (dirpath, dirnames,_) in os.walk(base_image_dir):
		if len(dirpath.split('/')) == 9:
			image_dirs += [os.path.join(dirpath)]

	features = load_json(os.path.join(base_project_dir, 'configs/features.json'))[feature_type]

	if table_video:
		print("making feature table for each trial")
		if len(glob.glob(os.path.join(base_project_dir, 'visualization', 'tables', 'trials', '*'))) <= 1:
			make_features_table(base_features_dir, users, base_project_dir, 'trials')

	for image_directory in image_dirs:
		shared_directory = os.path.join(image_directory.split("/")[-2], image_directory.split("/")[-1]) # the type of phrase and trial: phrase/trial_number

		image_directory_filepath = os.path.join(base_image_dir, shared_directory) # path to png images of this trial
		feature_directory_filepath = glob.glob(os.path.join(base_features_dir, shared_directory, '*.data'))[0] # path to raw mediapipe data 
		save_directory_filepath = os.path.join(base_project_dir, 'visualization/videos', shared_directory) # path to where the images/video should be saved

		print("Processing {}".format(save_directory_filepath))

		#print(shared_directory, image_directory_filepath, feature_directory_filepath, save_directory_filepath)

		if not os.path.exists(save_directory_filepath):
			os.makedirs(save_directory_filepath)
			print("Making Directory {}".format(save_directory_filepath))
			make_mediapipe_video(image_directory_filepath, feature_directory_filepath, save_directory_filepath, features, shared_directory, table_video, video_type_lists, frame_rate)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--base_image_dir', default = '/home/thad/Desktop/AndroidCaptureApp/DATA/Prerna_04_07_20/')
	parser.add_argument('--base_features_dir', default = '/media/thad/Seagate Backup Plus Drive/July_Mediapipe_Features')
	parser.add_argument('--users', default = ['Prerna'])
	parser.add_argument('--base_project_dir', default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs')
	parser.add_argument('--feature_type', default = 'visualization_features')
	parser.add_argument('--table_video', action = 'store_true')
	parser.add_argument('--video_type_lists', default = [['mediapipe']])
	parser.add_argument('--frame_rate', default = 5)
	args = parser.parse_args()

	make_visualization_videos(args.base_image_dir, args.base_features_dir, args.users, args.base_project_dir, args.feature_type, args.table_video, args.video_type_lists, args.frame_rate)
