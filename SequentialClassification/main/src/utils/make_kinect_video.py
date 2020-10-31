import numpy as np
import cv2 as cv
import json
# from matplotlib import pyplot as plt

def convert_points_2d(joint_positions, intrinsic):
	print(joint_positions.shape)
	joint_positions = joint_positions.transpose()
	joint_positions = joint_positions/joint_positions[-1]
	rgb = intrinsic @ joint_positions
	rgb = rgb[0:-1, :]
	rgb = rgb.transpose()
	return rgb


def make_kinect_video(video_filepath, features_filepath):

	with open(features_filepath) as f:
		data = json.load(f)

	cap = cv.VideoCapture(video_filepath)

	intrinsic = np.array([[900, 0, 940],[0, 900, 560], [0, 0, 1]])
	frameNumber = -1

	while cap.isOpened():
		ret, frame = cap.read()
		frameNumber += 1

		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break

		joint_positions = data["frames"][frameNumber]["bodies"][0]["joint_positions"]
		points = convert_points_2d(np.asarray(joint_positions), intrinsic)
		points = points.astype(int)
		
		for point in points:
			cv.circle(frame,tuple(point), 5, (0,255,0), -1)

		cv.circle(frame,tuple(points[8]), 8, (255,0,0), -1)
		cv.circle(frame,tuple(points[15]), 8, (0,0,255), -1)

		cv.imshow('frame', frame)

		if cv.waitKey(1) == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()		

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	parser.add_argument('--video_filepath', type = str)
	parser.add_argument('--features_filepath', type = str)
	args = parser.parse_args()

	make_kinect_video(args.video_filepath, args.features_filepath)

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