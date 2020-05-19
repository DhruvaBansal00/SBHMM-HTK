# MASTER PYTHON SCRIPT FOR THE COMPLETE VERIFICATION OR RECOGNITION PIPELINE
# Aditya Vishwanath

# Argument 1: Name of the feature set (e.g. 'tip_thumb_hand_dist')
# Argument 2: Number of dimensions of the feature set (e.g. '16')
# Argument 3: Recognition or Verification (0 is Recognition; 1 is Verification)

import os
import sys
import subprocess

feature_set = sys.argv[1]
num_dimensions = sys.argv[2]
choice = sys.argv[3]

# Creates all the following lists:
# The master MLF file for all sentences
# MLF files for each sentence as the ground truth
# Master training and testing lists
# Testing lists for each individual sentence
# List of all sentences	
#cmd = 'python data/{}/create_data_lists.py {}'.format(feature_set, feature_set)
cmd = 'python create_data_lists.py {}'.format(feature_set)
subprocess.call([cmd], shell=True)
# subprocess.call(["python3", "feature_extraction_mediapipe.py", "--input_filepath", raw_data_file,
# 				"--output_filepath", ark_file, "--feature_indexes", args.feature_indexes, "--three_dim"])
#os.system('python data/'+feature_set+'/create_data_lists.py '+feature_set)

# Move all flies to the appropriate folder
# os.system('rm -rf lists/')
# os.system('mkdir lists')
# os.system('mv '+feature_set+'.all ' 'lists/')
# os.system('mv '+feature_set+'.train ' 'lists/')
# os.system('mv '+feature_set+'_all.test ' 'lists/')

# Run the training script which creates the HMMs from the data
cmd = 'bash training_script.sh {} {}'.format(feature_set, num_dimensions)
subprocess.call([cmd], shell=True)
#os.system('bash training_script.sh '+feature_set+' '+num_dimensions)


if choice == '0':
	# Pure recognition
	os.system('mkdir results/'+feature_set+'_all')
	os.system('mkdir hresults/'+feature_set+'_all')
	os.system('bash testing_script_pure_recognition.sh '+feature_set+' '+num_dimensions+' '+feature_set+'_all')


if choice == '1':
	# Verification on all sentences

	old_sentence_list = ["Alligator_behind_black_wall","Alligator_behind_blue_wagon","Alligator_behind_chair","Alligator_behind_orange_wagon","Alligator_behind_wall","Alligator_in_box","Alligator_in_orange_flowers","Alligator_in_wagon","Alligator_on_bed","Alligator_on_blue_wall","Alligator_under_green_bed","Black_Alligator_behind_orange_wagon","Black_cat_behind_green_bed","Black_cat_in_blue_wagon","Black_cat_on_green_bed","Black_Snake_under_blue_chair","Black_Spider_in_white_flowers","Blue_Alligator_on_green_wall","Blue_Spider_on_green_box","cat_behind_orange_bed","Cat_behind_bed","Cat_behind_box","Cat_behind_flowers","Cat_on_blue_bed","Cat_on_green_wall","Cat_on_wall","Cat_under_blue_bed","Cat_under_chair","cat_under_orange_chair","Green_Alligator_under_blue_flowers","Green_Snake_under_blue_chair","Green_snake_under_blue_chair","Green_Spider_under_orange_chair","Orange_Alligator_in_green_flowers","Orange_Snake_under_blue_flowers","Orange_Spider_in_green_box","Orange_spider_under_green_flowers","Snake_behind_wall","Snake_in_flowers","Snake_in_green_wagon","Snake_on_box","Snake_under_bed","Snake_under_black_chair","Snake_under_blue_chair","Snake_under_blue_flowers","Snake_under_chair","Spider_under_bed","Spider_in_blue_box","Spider_in_box","Spider_in_green_box","Spider_in_orange_flowers","Spider_on_chair","Spider_on_wall","Spider_on_white_wall","Spider_under_blue_chair","Spider_under_wagon","White_snake_in_blue_flowers","White_Alligator_on_blue_wall","White_cat_in_green_box","White_cat_on_orange_wall"]
	sentence_list = []
	for sentence in old_sentence_list:
		sentence = sentence.lower()
		# sentence = sentence.replace('_on', '_above')
		# sentence = sentence.replace('spider', 'monkey')
		# sentence = sentence.replace('behind', 'above')
		# sentence = sentence.replace('under', 'below')
		# sentence = sentence.replace('cat', 'lion')
		# sentence = sentence.replace('green', 'grey')
		sentence_list.append(sentence)

	for ground_truth in sentence_list: # This is the ground truth
		# Set up the ground truth folders
		os.system('rm -rf results/'+ground_truth)
		os.system('mkdir results/'+ground_truth)
		os.system('rm -rf hresults/'+ground_truth)
		os.system('mkdir hresults/'+ground_truth)
		for incoming_sentence in sentence_list:
			os.system('mkdir results/'+ground_truth+'/'+incoming_sentence)
			os.system('mkdir hresults/'+ground_truth+'/'+incoming_sentence)
			# Run verification
			os.system('bash testing_script.sh '+feature_set+' '+num_dimensions+' '+incoming_sentence+' '+ground_truth)