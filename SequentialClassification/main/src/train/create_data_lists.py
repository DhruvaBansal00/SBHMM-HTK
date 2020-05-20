"""Creates train.data, test.data, and all.data, which contain the
phrases in the dataset.

Methods
-------
create_data_lists
"""
import os
import sys
import glob
import argparse
from random import shuffle


def create_data_lists(
		train_data: list, test_data: list, users: list = [], 
		phrase_len: int = 0, test_on_train: bool = False) -> None:
	"""Creates train.data, test.data, and all.data, which contain the
	phrases in the dataset.

	Parameters
	----------
	train_data : list
		All phrases to be included as training data.

	test_data : list
		All phrases to be included as testing data.

	users : list, optional, by default []
		Only include specified users.

	phrase_len : int, optional, by default 0
		Only include phrases of the specified length.

	test_on_train : bool, optional, by default False
		If all phrases are used for training and testing.
	"""

	htk_dir = os.path.join('data', 'htk')
	
	if not os.path.exists('lists'):
		os.makedirs('lists')

	if len(users) > 0:
		filenames = []
		for user in users:
			htk_filepaths = os.path.join(htk_dir, '*{}*.htk'.format(user))
			filenames += glob.glob(htk_filepaths)

	else:
		htk_filepaths = os.path.join(htk_dir, '*.htk')
		filenames = glob.glob(htk_filepaths)

	all_data_filepath = os.path.join('lists', 'all.data')
	train_data_filepath = os.path.join('lists', 'train.data')
	test_data_filepath = os.path.join('lists', 'test.data')

	with open(all_data_filepath, 'w') as all_data_list, \
		 open(train_data_filepath, 'w') as train_data_list, \
		 open(test_data_filepath, 'w') as test_data_list:

		if test_on_train:

			for phrase in train_data:
				all_data_list.write('{}\n'.format(phrase))
				train_data_list.write('{}\n'.format(phrase))
				test_data_list.write('{}\n'.format(phrase))

		else:

			for phrase in train_data:
				all_data_list.write('{}\n'.format(phrase))
				train_data_list.write('{}\n'.format(phrase))

			for phrase in test_data:
				all_data_list.write('{}\n'.format(phrase))
				test_data_list.write('{}\n'.format(phrase))