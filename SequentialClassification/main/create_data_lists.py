# Written by Aditya Vishwanath

# Argument 1: Feature set name

#python3 create_data_lists.py --feature_set_name Mediapipe --phrase_len 3 --users ravi


import os
import sys
import glob
import argparse
from random import shuffle

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--feature_set_name', type=str, required=True, help='Name of feature set being used for saving lists')
	parser.add_argument('--users', type=str, nargs='*', default=[], help='Names of specific users to include')
	parser.add_argument('--phrase_len', type=int, help='Use only phrases of specified length')
	args = parser.parse_args()

	print(args)

	if len(args.users) > 0:
		filenames = []
		for user in args.users:
			filenames += glob.glob('data/{}/htk/*{}*.htk'.format(args.feature_set_name, user))

	else:
		filenames += glob.glob('data/{}/htk/*.htk'.format(args.feature_set_name))

	filenames_list = []
	filenames_list_2 = []
	unique_phrases = set()

	print()

	# with open('lists/{}.all'.format(args.feature_set_name), 'w') as in_files, \
	# 	 open('all_labels.mlf', 'w+') as mlf_files:
	with open('lists/{}.all'.format(args.feature_set_name), 'w') as in_files:
		for eachfile in filenames:
			if not args.phrase_len or args.phrase_len == len(eachfile.split('/')[-1].split('.')[0].split('_')) - 2:
				unique_phrases.add(' '.join(eachfile.split('/')[-1].split('.')[0].split('_')[1:-1]))
				eachfile = eachfile.replace(os.getcwd(), 'data/{}'.format(args.feature_set_name))
				eachfile = eachfile.replace('data/'+args.feature_set_name+'/data/'+args.feature_set_name+'/', 'data/'+args.feature_set_name+'/') # ADDITIONAL FIX IF CALLING THIS FILE FROM THE RUN.PY MASTER FILE
				filenames_list.append(eachfile+'\n')
				filenames_list_2.append(eachfile)
				in_files.write(eachfile + '\n')


	for phrase in unique_phrases:
		print(phrase)
	# Creating the global MLF File 

	f = open('all_labels.mlf', 'w+')
	f.write("#!MLF!#\n")
	for name in filenames_list_2:
		name = name.replace('data/' + args.feature_set_name + '/htk/', '*/')
		# name = name.replace('data/tip_thumb_hand_dist/*/', '*/') #ALSO REMOVE THIS LINE IF YOU ARE EXECUTING THIS FILE FROM THE PRESENT DIR	
		name = name.replace('.htk', '.lab')
		f.write("\""+name+"\"\n")
		words = name.split('_')
		f.write("sil0\n")
		for x in range(1, len(words) - 1):
			f.write((words[x]).lower() + '\n')
			# if words[x].lower() == 'on':
			# 	f.write('above\n')
			# elif words[x].lower() == 'spider':
			# 	f.write('monkey\n')
			# elif words[x].lower() == 'behind':
			# 	f.write('above\n')
			# elif words[x].lower() == 'under':
			# 	f.write('below\n')
			# elif words[x].lower() == 'cat':
			# 	f.write('lion\n')
			# elif words[x].lower() == 'green':
			# 	f.write('grey\n')
			# else:
			# 	f.write((words[x]).lower() + '\n')
		f.write("sil1\n")
		f.write('.\n')


	# Training and testing lists
	# For creating testing and training lists with 10 users for training and 2 users for testing
	# This will create both the super testing list AND also testing lists for each individual sentence

	sentence_list = ["Alligator_behind_black_wall","Alligator_behind_blue_wagon","Alligator_behind_chair","Alligator_behind_orange_wagon","Alligator_behind_wall","Alligator_in_box","Alligator_in_orange_flowers","Alligator_in_wagon","Alligator_on_bed","Alligator_on_blue_wall","Alligator_under_green_bed","Black_Alligator_behind_orange_wagon","Black_cat_behind_green_bed","Black_cat_in_blue_wagon","Black_cat_on_green_bed","Black_Snake_under_blue_chair","Black_Spider_in_white_flowers","Blue_Alligator_on_green_wall","Blue_Spider_on_green_box","cat_behind_orange_bed","Cat_behind_bed","Cat_behind_box","Cat_behind_flowers","Cat_on_blue_bed","Cat_on_green_wall","Cat_on_wall","Cat_under_blue_bed","Cat_under_chair","cat_under_orange_chair","Green_Alligator_under_blue_flowers","Green_Snake_under_blue_chair","Green_snake_under_blue_chair","Green_Spider_under_orange_chair","Orange_Alligator_in_green_flowers","Orange_Snake_under_blue_flowers","Orange_Spider_in_green_box","Orange_spider_under_green_flowers","Snake_behind_wall","Snake_in_flowers","Snake_in_green_wagon","Snake_on_box","Snake_under_bed","Snake_under_black_chair","Snake_under_blue_chair","Snake_under_blue_flowers","Snake_under_chair","Spider_under_bed","Spider_in_blue_box","Spider_in_box","Spider_in_green_box","Spider_in_orange_flowers","Spider_on_chair","Spider_on_wall","Spider_on_white_wall","Spider_under_blue_chair","Spider_under_wagon","White_snake_in_blue_flowers","White_Alligator_on_blue_wall","White_cat_in_green_box","White_cat_on_orange_wall"]
	#sentence_list.sort(lambda x,y: -cmp(len(x.split('_')), len(y.split('_'))))
	sentence_list.sort(key=lambda x: len(x))

	shuffle(filenames_list)
	filenames_list_3 = filenames_list[:]
	os.system('rm -rf test_file_lists')
	os.system('mkdir test_file_lists')

	train_list_file = open('lists/' + args.feature_set_name+'.train', 'w+')
	super_test_list_file = open('lists/' + args.feature_set_name+'_all.test', 'w+')

	trainOnTest = True

	for sentence in sentence_list:
		c = 0
		sentence2 = sentence
		sentence2 = sentence2.lower()
		# sentence2 = sentence2.replace('_on', '_above')
		# sentence2 = sentence2.replace('spider', 'monkey')
		# sentence2 = sentence2.replace('behind', 'above')
		# sentence2 = sentence2.replace('under', 'below')
		# sentence2 = sentence2.replace('cat', 'lion')
		# sentence2 = sentence2.replace('green', 'grey')
		test_list_file = open('test_file_lists/'+sentence2+'.test', 'w+')
		filenames_list_copy = filenames_list[:]
		for line in filenames_list:

			if sentence in line:
				filenames_list_copy.remove(line)
				if not trainOnTest:
					if ('ravi' in line) or ('kartavya' in line):
					# if ('kartavya' in line) or ('nikhil' in line) or ('kmd' in line) or ('pranathi' in line): #CHANGE NAME HERE!!!
						test_list_file.write(line)
						super_test_list_file.write(line)
					else:
						train_list_file.write(line)
				else:
					test_list_file.write(line)
					super_test_list_file.write(line)
					train_list_file.write(line)
		filenames_list = filenames_list_copy[:]


	# For creating all MLF files for all sentences as the ground truth

	os.system('rm -rf mlf_files')
	os.system('mkdir mlf_files')
	for sentence in sentence_list:
		content = []
		with open('lists/' + args.feature_set_name+'.all') as all_test_files:
			content = all_test_files.readlines()
		sentence2 = sentence
		sentence2 = sentence2.lower()
		# sentence2 = sentence2.replace('_on', '_above')
		# sentence2 = sentence2.replace('spider', 'monkey')
		# sentence2 = sentence2.replace('behind', 'above')
		# sentence2 = sentence2.replace('under', 'below')
		# sentence2 = sentence2.replace('cat', 'lion')
		# sentence2 = sentence2.replace('green', 'grey')
		f2 = open('mlf_files/' + sentence2 + '_test_labels.mlf', 'w+')
		f2.write("#!MLF!#\n")
		for line in content:
			line = line[:].strip("\n")
			line = line.replace('data/' + args.feature_set_name + '/htk/', '*/')
			# line = line.replace('data/tip_thumb_hand_dist/*/', '*/') #ALSO REMOVE THIS LINE IF YOU ARE EXECUTING THIS FILE FROM THE PRESENT DIR	
			line = line.replace('.htk', '.lab')
			f2.write("\""+line+"\"\n")
			words = sentence.lower().split('_')
			f2.write("sil0\n")
			for word in words:
				f2.write(word.lower() + '\n')
				# if word == 'on':
				# 	f2.write('above\n')
				# elif word == 'spider':
				# 	f2.write('monkey\n')
				# elif word == 'behind':
				# 	f2.write('above\n')
				# elif word == 'under':
				# 	f2.write('below\n')
				# elif word == 'cat':
				# 	f2.write('lion\n')
				# elif word == 'green':
				# 	f2.write('grey\n')
				# else:
				# 	f2.write(word.lower() + '\n')
			f2.write("sil1\n")
			f2.write('.\n')
		f2.close()