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


def create_data_lists(train_data: list, test_data: list, phrase_len: int = 0) -> None:
	"""Creates train.data, test.data, and all.data, which contain the
	phrases in the dataset.

	Parameters
	----------
	train_data : list
		All phrases to be included as training data.

	test_data : list
		All phrases to be included as testing data.

	phrase_len : int, optional, by default 0
		Only include phrases of the specified length.
	"""
	#TODO: Add filter for phrase length

	if not os.path.exists('lists'):
		os.mkdir('lists')

	all_data_filepath = os.path.join('lists', 'all.data')
	train_data_filepath = os.path.join('lists', 'train.data')
	test_data_filepath = os.path.join('lists', 'test.data')

	with open(all_data_filepath, 'w') as all_data_list, \
		 open(train_data_filepath, 'w') as train_data_list, \
		 open(test_data_filepath, 'w') as test_data_list:

			for phrase in train_data:
				all_data_list.write('{}\n'.format(phrase))
				train_data_list.write('{}\n'.format(phrase))

			for phrase in test_data:
				all_data_list.write('{}\n'.format(phrase))
				test_data_list.write('{}\n'.format(phrase))