import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import os
from get_confusion_matrix import get_confusion_matrix
from plot_macros_gaussian import get_macros
from json_data import load_json


def gaussian(x, mu, var):
    """Calculates the output value y for a given guassian and input value x using the following formula:
        f(x) = 1 / (sqrt(2 * PI * variance)) * E^(-((x - mu)^2 / (2 * variance))) .

    Parameters
    ----------
    x : float
        The input x value to the function.

    mu : float
        The mean of the guassian.

    var : float
        The variance of the guassian.

    Returns
    -------
    y : float
        The output y value from the function defined above.

    """
    return (1 / (np.sqrt(2 * np.pi * var))) * (np.power(np.e, -(np.power((x - mu), 2) / (2 * var))))

def calculate_bhattacharyya_distance(macros_data, save_dir, words, feature_label):
    """Calculates the Bhattacharyya Distance for a specific feature label of the two confused words.
    Compares mixture gaussian model of (word[0] feature_label state_number_i) with (word[1] feature_label state_number_i) for each i in range(num_states).


    Parameters
    ----------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label].

    save_dir : str
        The directory where the file with Bhattacharyya distances will be saved too.

    words : list of str
        A list of filtered sign words to generate data on (if some pair of words exceed the threshold).

    feature_labels : list of str
        A list of features to generate data on.

    Returns
    -------
    bhattacharyya_distance : float
        The output bhattacharyya_distance value for the specific feature label of the two confused words.

    """

def find_confused_word(macros_data, save_dir, words, feature_labels, confusion_matrix_filepath, threshold):
    """Prebuilt function that finds pair of confused words that exceeds the threshold from the confusion matrix
        and calls calculate_bhattacharyya_distance on all the feature labels. 

    Parameters
    ----------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label].

    save_dir : str
        The directory where the file with Bhattacharyya distances will be saved too.

    words : list of str
        A list of filtered sign words to generate data on (if some pair of words exceed the threshold).

    feature_labels : list of str
        A list of features to generate data on.
    
    confusion_matrix_filepath : str
        File path to the confusion matrix file that is generated from testing HMM.

    threshold : float
        If the percentage of times a word is mislabeled exceeds the threshold, then the two words are confused (ground_truth, HMM classification).

    Returns
    -------
    None
        On success, a file with the Bhattacharyya Distance for each unique pair of confused words and feature_label is generated in the specified save directory.
        Depending on the threshold and confusion matrix, the file may be empty.

    """

    # confusion_matrix_dict: [ground_truth_word (vertical_axis_of_confusion_matrix)][predicted_word (horizontal_axis_of_confusion_matrix)].
    confusion_matrix_dict = get_confusion_matrix(confusion_matrix_filepath)['matrix']

    for row in confusion_matrix_dict.keys():
        total = sum(confusion_matrix_dict[row].values())
        if not total: continue

        for col in confusion_matrix_dict[row]:
            if not (row in col or col in row) and confusion_matrix_dict[row][col] / float(total) >= float(threshold):
                row_word = None
                col_word = None
                for word in words:
                    if row in word: row_word = word
                    if col in word: col_word = word
                
                if not row_word or not col_word: continue

                word_dir = os.path.join(save_dir, '_'.join([row_word, col_word]))
                if not os.path.exists(word_dir):
                    os.makedirs(word_dir)    

                for feature_label in feature_labels:
                    print("Bhattacharyya Distance for confused words ({}, {}) for feature {}".format(row_word, col_word, feature_label))
                    calculate_bhattacharyya_distance(macros_data, word_dir, [row_word, col_word], feature_label)

def bhattacharyya_distance(feature_config_filepath, feature_config_key, macros_filepath, save_dir, words, feature_labels, confusion_matrix_filepath, threshold, mode):
    """Function that calculates Bhattacharyya Distance using newMacros data.

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that consists of a json dictionary with different lists of features.

    feature_config_key : str
        The features that were selected from the feature_config_filepath when testing and training the HMM.

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM.

    save_dir : str
        The directory where the file with Bhattacharyya distances will be saved too.

    words : list of str
        A list of filtered sign words to generate data on (if some pair of words exceed the threshold).
        If empty list, then all words from the newMacros file are utilized.

    feature_labels : list of str
        A list of features to generate data on.
        If empty list, then all feature labels from feature_config are utilized.

    confusion_matrix_filepath : str
        File path to the confusion matrix file that is generated from testing HMM.

    threshold : float
        If the percentage of times a word is mislabeled exceeds the threshold,
        then the two words are confused (ground truth, HMM predicted classification).

    NOT USED CURRENTLY:
    mode : int
        If mode == 0, then plot each word and each feature seperately.
        If mode == 1, then plot each confused pair of words together for each feature.

    Returns
    -------
    None
        On success, generates a file that consists of good features from the Bhattacharyya Distance.

    """
    # gaussian_dir = os.path.join(save_dir, 'visualization', 'gaussian', str(mode))
    # if os.path.exists(gaussian_dir):
    #     shutil.rmtree(gaussian_dir)
    # os.makedirs(gaussian_dir)

    all_features = load_json(feature_config_filepath)[str(feature_config_key)]
    macros_data = get_macros(all_features, macros_filepath)

    if not feature_labels:
        feature_labels = all_features

    if not words:
        words = macros_data.keys()

    find_confused_word(feature_config_filepath, macros_filepath, gaussian_dir, feature_labels, confusion_matrix_filepath, threshold)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_config_filepath', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs/configs/features.json')
    parser.add_argument('--feature_config_key', type = str, default = 'selected_features')
    parser.add_argument('--macros_filepath', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs/models/hmm80/newMacros')
    parser.add_argument('--save_dir', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs/')
    parser.add_argument('--words', nargs='*', type = str, default = [])
    parser.add_argument('--feature_labels', nargs='*', type = str, default = [])
    parser.add_argument('--confusion_matrix_filepath', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs/hresults/res_hmm80.txt')
    parser.add_argument('--threshold', type = float, default = 0.05)
    parser.add_argument('--mode', type = int, default = 0)
    args = parser.parse_args()

    """Calculate Bhattacharyya distance between words for mixture model gaussians to determine good features.

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that consists of a json dictionary with different lists of features.

    feature_config_key : str
        The features that were selected from the feature_config_filepath when testing and training the HMM.

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM.

    save_dir : str
        The directory where the file with Bhattacharyya distances will be saved too.

    words : list of str
        A list of filtered sign words to generate data on (if some pair of words exceed the threshold).
        If empty list, then all words from the newMacros file are utilized.

    feature_labels : list of str
        A list of features to generate data on.
        If empty list, then all feature labels from feature_config are utilized.

    confusion_matrix_filepath : str
        File path to the confusion matrix file that is generated from testing HMM.

    threshold : float
        If the percentage of times a word is mislabeled exceeds the threshold,
        then the two words are confused (ground truth, HMM predicted classification).

    NOT USED CURRENTLY:
    mode : int
        If mode == 0, then plot each word and each feature seperately.
        If mode == 1, then plot each confused pair of words together for each feature.

    """
    bhattacharyya_distance(args.feature_config_filepath, args.feature_config_key, args.macros_filepath, args.save_dir, args.words, args.feature_labels, args.confusion_matrix_filepath, args.threshold, args.mode)