import numpy as np
import cv2 as cv
import json
import os
import argparse
import glob
# from matplotlib import pyplot as plt

def convert_points_2d(joint_positions, intrinsic):
	# print(joint_positions.shape)
	joint_positions = joint_positions.transpose()
	joint_positions = joint_positions/joint_positions[-1]
	rgb = intrinsic @ joint_positions
	rgb = rgb[0:-1, :]
	rgb = rgb.transpose()
	return rgb

def make_kinect_video(video_filepath, output_project_directory, features_filepath):

	filename = os.path.split(video_filepath)[1]
	session, phrase, trial, _ = filename.split('.')
	save_directory = os.path.join(output_project_directory, 'visualization', 'videos', session, phrase, trial)

	if not os.path.exists(save_directory):
		os.makedirs(save_directory)

	num_frames = draw_features_kinect(video_filepath, features_filepath, save_directory)
	save_video_kinect(save_directory, filename, 5)
	delete_images_kinect(save_directory, num_frames)

def delete_images_kinect(save_directory, num_frames):
	file_location = os.path.join(save_directory, 'frame')
	for i in range(num_frames):
		filename = f'{file_location}_{i:04d}.png'
		os.remove(filename)

def save_video_kinect(save_directory, video_filename, fps):

	session, phrase, trial = video_filename.split('.')[0:3]
	video_filename = '.'.join((session, phrase, trial, 'mp4'))

	frame_size = (3840, 2160)
	out = cv.VideoWriter(os.path.join(save_directory, video_filename), cv.VideoWriter_fourcc(*'MP4V'), fps, frame_size)
	for filename in sorted(glob.glob(os.path.join(save_directory, '*.png'))):
		img = cv.imread(filename)
		out.write(img)

	out.release()

def draw_features_kinect(video_filepath, features_filepath, save_directory):

	with open(features_filepath) as f:
		data = json.load(f)

	vidcap = cv.VideoCapture(video_filepath)

	success, image = vidcap.read()
	frameNumber = 0

	intrinsic = np.array([[900, 0, 940],[0, 900, 560], [0, 0, 1]])

	while success:
		framename = f'frame_{frameNumber:04d}.png'

		joint_positions = data["frames"][frameNumber]["bodies"][0]["joint_positions"]
		points = convert_points_2d(np.asarray(joint_positions), intrinsic)
		points = points.astype(int)
		
		for point in points:
			cv.circle(image, tuple(point), 5, (0,255,0), -1)

		cv.circle(image, tuple(points[8]), 8, (255,0,0), -1) # left hand
		cv.circle(image, tuple(points[15]), 8, (0,0,255), -1) # right hand

		new_frame_filepath = os.path.join(save_directory, framename)
		cv.imwrite(new_frame_filepath, image)

		frameNumber += 1
		success, image = vidcap.read()

	return frameNumber
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_filepath', type = str, default = '/media/thad/U32 Shadow/Video_Backup_With_Features_DO_NOT_DELETE/DATA/Videos')
	parser.add_argument('--output_project_directory', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Kinect', help = 'Directory path to where the video(s) will be saved.')	
	parser.add_argument('--features_filepath', type = str, default = '/media/thad/disk1/ProcessingPipeline/DATA/Kinect_Data_July_2020')
	args = parser.parse_args()

	filename = '08-13-20_Thad_4K.alligator_above_bed.0000000000'

	session, phrase, trial = filename.split('.')

	video_filepath = os.path.join(args.video_filepath, session, phrase, trial, f'{filename}.mkv')
	features_filepath = os.path.join(args.features_filepath, f'{filename}.json')

	print(video_filepath)

	make_kinect_video(video_filepath, args.output_project_directory, features_filepath)

# # cap = cv.VideoCapture('alligator_above_chair.1.mkv')
# cap = cv.VideoCapture('black_monkey_in_white_flowers.1.605.mkv')
# # intrinsic = np.array([[900, 0, 960],[0, 900, 540], [0, 0, 1]])
# intrinsic = np.array([[900, 0, 940],[0, 900, 560], [0, 0, 1]])
# frameNumber = -1;
# while cap.isOpened():
# 	ret, frame = cap.read()
# 	frameNumber = frameNumber + 1
# 	# plt.imshow(frame)
# 	# plt.show()
# 	# print(frame.shape)
# 	if not ret:
# 		print("Can't receive frame (stream end?). Exiting ...")
# 		break
# 	joint_positions = data["frames"][frameNumber]["bodies"][0]["joint_positions"]
# 	points = convert_points_2d(np.asarray(joint_positions), intrinsic)
# 	points = points.astype(int)
# 	for point in points:
# 		cv.circle(frame,tuple(point), 5, (0,255,0), -1)
# 	cv.circle(frame,tuple(points[8]), 8, (255,0,0), -1)
# 	cv.circle(frame,tuple(points[15]), 8, (0,0,255), -1)
# 	cv.imshow('frame', frame)
# 	if cv.waitKey(1) == ord('q'):
# 		break
# cap.release()
# cv.destroyAllWindows()