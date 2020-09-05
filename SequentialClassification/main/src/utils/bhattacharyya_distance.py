import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import json
import os
from get_confusion_matrix import get_confusion_matrix
from plot_macros_gaussian import get_macros
from json_data import load_json
from scipy.integrate import quad


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

def integrand(x, mu_list1, var_list1, weight_list1, mu_list2, var_list2, weight_list2):
    gaussian_sum1 = 0
    for i in range(len(mu_list1)):
        gaussian_sum1 += weight_list1[i] * gaussian(x, mu_list1[i], var_list1[i])
    gaussian_sum2 = 0
    for i in range(len(mu_list1)):
        gaussian_sum2 += weight_list2[i] * gaussian(x, mu_list2[i], var_list2[i])
    return np.power(gaussian_sum1 * gaussian_sum2, 0.5)

def bhattacharyya(integrand, mu_list1, var_list1, weight_list1, mu_list2, var_list2, weight_list2, lower, upper):
    return quad(integrand, lower, upper, args=(mu_list1, var_list1, weight_list1, mu_list2, var_list2, weight_list2))

def calculate_bhattacharyya_distance(macros_data, words, feature_label):
    """Calculates the Bhattacharyya Distance for a specific feature label of the two confused words.
    Compares mixture gaussian model of (word[0] feature_label state_number_i) with (word[1] feature_label state_number_i) for each i in range(num_states).


    Parameters
    ----------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label].

    words : list of str
        A list of filtered sign words to generate data on (if some pair of words exceed the threshold).

    feature_labels : list of str
        A list of features to generate data on.

    Returns
    -------
    bhatt_dist_list : list of float
        The output bhattacharyya distance values for the specific feature label of the two confused words for each state.

    """

    bhatt_dist_list = []

    for state in range(2, min(len(macros_data[words[0]]), len(macros_data[words[1]])) + 2):
        mu_list1 = [ macros_data[words[0]][state][mix]['mean'][feature_label] for mix in macros_data[words[0]][state].keys() ]
        var_list1 = [ macros_data[words[0]][state][mix]['variance'][feature_label] for mix in macros_data[words[0]][state].keys() ]
        weight_list1 = [ macros_data[words[0]][state][mix]['mixture_weight'] for mix in macros_data[words[0]][state].keys() ]

        mu_list2 = [ macros_data[words[1]][state][mix]['mean'][feature_label] for mix in macros_data[words[1]][state].keys() ]
        var_list2 = [ macros_data[words[1]][state][mix]['variance'][feature_label] for mix in macros_data[words[1]][state].keys() ]
        weight_list2 = [ macros_data[words[1]][state][mix]['mixture_weight'] for mix in macros_data[words[1]][state].keys() ]

        #print(var_list1)
        bhatt_dist_value = bhattacharyya(integrand, mu_list1, var_list1, weight_list1, mu_list2, var_list2, weight_list2, -20.0, 20.0)[0]
        bhatt_dist_value = round(bhatt_dist_value, 3)
        bhatt_dist_list.append(bhatt_dist_value)

    return bhatt_dist_list

def find_confused_word(macros_data, words, feature_labels, confusion_matrix_filepath, threshold):
    """Prebuilt function that finds pair of confused words that exceeds the threshold from the confusion matrix
        and calls calculate_bhattacharyya_distance on all the feature labels. 

    Parameters
    ----------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label].

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
    bhatt_dist_dict : dictionary
        The dictionary for all confused words for the specific feature label for each state.
        Depending on the threshold and confusion matrix, the dictionary may be empty.
        [str(word[0]_word[1])][feature_label][state_number/average].

    """

    # confusion_matrix_dict: [ground_truth_word (vertical_axis_of_confusion_matrix)][predicted_word (horizontal_axis_of_confusion_matrix)].
    confusion_matrix_dict = get_confusion_matrix(confusion_matrix_filepath)['matrix']

    bhatt_dist_dict = {}

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

                confused_words = '_'.join([row_word, col_word]) 

                bhatt_dist_dict[confused_words] = {}

                for feature_label in feature_labels:
                    print("Bhattacharyya Distance for confused words ({}, {}) for feature {}".format(row_word, col_word, feature_label))
                    bhatt_dist_dict[confused_words][feature_label] = {}
                    bhatt_dist_list = calculate_bhattacharyya_distance(macros_data, [row_word, col_word], feature_label)

                    for state_num, bhatt_dist in enumerate(bhatt_dist_list): 
                        bhatt_dist_dict[confused_words][feature_label][state_num + 2] = bhatt_dist

                    bhatt_dist_dict[confused_words][feature_label]['average'] = round(sum(bhatt_dist_list) / float(len(bhatt_dist_list)), 3)

    return bhatt_dist_dict

def bhattacharyya_distance(split, feature_config_filepath, feature_config_key, macros_filepath, save_dir, words, feature_labels, confusion_matrix_filepath, threshold, mode):
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
    gaussian_dir = os.path.join(save_dir, 'visualization', 'gaussian', '1', split)
    if not os.path.exists(gaussian_dir):
        os.makedirs(gaussian_dir)

    all_features = load_json(feature_config_filepath)[str(feature_config_key)]
    macros_data = get_macros(all_features, macros_filepath)

    if not feature_labels:
        feature_labels = all_features

    if not words:
        words = macros_data.keys()

    bhatt_dist_dict = find_confused_word(macros_data, words, feature_labels, confusion_matrix_filepath, threshold)

    feature_dict = {}

    for feature_label in feature_labels:
        total = 0
        print(feature_label)
        for confused_word in bhatt_dist_dict.keys():
            print(confused_word)
            total += bhatt_dist_dict[confused_word][feature_label]['average']
        print(total, len(bhatt_dist_dict.keys()))
        if not total: continue
        feature_dict[feature_label] = round(total / float(len(bhatt_dist_dict.keys())), 3)


    confused_word_dict = {}

    for confused_word in bhatt_dist_dict.keys():
        confused_word_dict[confused_word] = sorted(bhatt_dist_dict[confused_word].items(), key = lambda kv:(kv[1]['average'], kv[0]))
    print(confused_word_dict)

    bhatt_dist_dict = confused_word_dict
    bhatt_dist_dict['features'] = feature_dict
    bhatt_dist_dict['features'] = sorted(bhatt_dist_dict['features'].items(), key = lambda kv:(kv[1], kv[0]))



    # write the bhatt score to a file in the respective directory
    bhatt_filepath = os.path.join(gaussian_dir, "bhatt_dist.json")
    with open(bhatt_filepath, "w") as file:
        file.write(json.dumps(bhatt_dist_dict, indent=4)) 

    #print(bhatt_dist_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_config_filepath', type = str, default = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/configs/features.json')
    parser.add_argument('--feature_config_key', type = str, default = 'selected_features')
    parser.add_argument('--macros_filepath', type = str, default = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/models/')
    parser.add_argument('--save_dir', type = str, default = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/')
    parser.add_argument('--words', nargs='*', type = str, default = [])
    parser.add_argument('--feature_labels', nargs='*', type = str, default = [])
    parser.add_argument('--confusion_matrix_filepath', type = str, default = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/hresults/')
    parser.add_argument('--threshold', type = float, default = 0.1)
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
    for i in range(7):
        macros_filepath = os.path.join(args.macros_filepath, str(i), 'hmm220/newMacros')
        confusion_matrix_filepath = os.path.join(args.confusion_matrix_filepath, str(i), 'res_hmm220.txt')
        bhattacharyya_distance(str(i), args.feature_config_filepath, args.feature_config_key, macros_filepath, args.save_dir, args.words, args.feature_labels, confusion_matrix_filepath, args.threshold, args.mode)
