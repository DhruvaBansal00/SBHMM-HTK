import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import os
from get_confusion_matrix import get_confusion_matrix

def gaussian(x, mu, var):
    """Calculates the output value y for a given guassian and input value x using the following formula:
        f(x) = 1 / (sqrt(2 * PI * variance)) * E^(-((x - mu)^2 / (2 * variance))) 

    Parameters
    ----------
    x : float
        The input x value to the function

    mu : float
        The mean of the guassian 

    var : float
        The variance of the guassian

    Returns
    -------
    y : float
        The output y value from the function defined above

    """
    return (1 / (np.sqrt(2 * np.pi * var))) * (np.power(np.e, -(np.power((x - mu), 2) / (2 * var))))


def get_macros(feature_config_filepath, macros_filepath):
    """Processes raw macros data extracted from HMMs and coverts the data into a dictionary macros_data:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/var then feature label]

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that lists the features used  when testing and training the HMM

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM

    Returns
    -------
    macros_data : dictionary
        The data extracted from the newMacros file in the following format:
        [word][state_number][mixture_number][mean/variance/gconst/mixture_weight][if mean/variance then feature_label]

    """
    macros_data = {}
    macro_lines = [ line.rstrip() for line in open(macros_filepath, "r") ]
    feature_labels = [ feature_label.rstrip() for feature_label in open(feature_config_filepath) ]

    i = 0
    while i != len(macro_lines):
        while i != len(macro_lines) and "~h" not in macro_lines[i]:
            i += 1
        if i == len(macro_lines): break
        word = macro_lines[i].split("\"")[1]
        macros_data[word] = {}
        
        while "<NUMSTATES>" not in macro_lines[i]:
            i += 1
        num_states = int(macro_lines[i].split(" ")[1])
        for num_state in range(2, num_states):
            while "<STATE>" not in macro_lines[i]:
                i += 1
            macros_data[word][num_state] = {}

            while "<NUMMIXES>" not in macro_lines[i]:
                i += 1
            num_mixes = int(macro_lines[i].split(" ")[1])
            for num_mix in range(1, num_mixes + 1):
                macros_data[word][num_state][num_mix] = {}

                i += 1
                macros_data[word][num_state][num_mix]["mixture_weight"] = float(macro_lines[i].split(" ")[2])

                i += 2
                mean_list = macro_lines[i].split(" ")[1:]
                mean_list = [ float(item) for item in mean_list ]
                macros_data[word][num_state][num_mix]["mean"] = dict(zip(feature_labels, mean_list))

                i += 2
                variance_list = macro_lines[i].split(" ")[1:]
                variance_list = [ float(item) for item in variance_list ]
                macros_data[word][num_state][num_mix]["variance"] = dict(zip(feature_labels, variance_list))

                i += 1
                macros_data[word][num_state][num_mix]["gconst"] = float(macro_lines[i].split(" ")[1])

        while "<ENDHMM>" not in macro_lines[i]:
            i += 1
    return macros_data

def generate_graph(feature_config_filepath, macros_filepath, save_dir, words, feature_label):
    """Generates a single graph consisting of guassian mixture model plot(s) for each word for each state

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that lists the features used  when testing and training the HMM

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM

    save_dir : str
        The directory where the graph plots are saved

    words : list of str
        A list of words that will be plotted on the same graph 

    feature_label : str
        The feature that the graph represents

    Returns
    -------
    None
        On success, a single graph is generated in the specified save directory

    """
    macros_data = get_macros(feature_config_filepath, macros_filepath)
    linestyles = ['-','--', ':', '-.']
    linecolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    words_str = ", ".join(words)

    fig, ax = plt.subplots(figsize=(9,6))
    fig_min = float("+inf")
    fig_max = float("-inf")
    gaussian_peaks = []
    for line, word in enumerate(words):

        for color, state in enumerate(macros_data[word]):
            state_min = float("+inf")
            state_max = float("-inf")
            for mixes in macros_data[word][state]:
                
                left_end = macros_data[word][state][mixes]["mean"][feature_label] - np.sqrt(macros_data[word][state][mixes]["variance"][feature_label]) * 3
                state_min = min(state_min, left_end)
                fig_min = min(fig_min, state_min)

                right_end = macros_data[word][state][mixes]["mean"][feature_label] + np.sqrt(macros_data[word][state][mixes]["variance"][feature_label]) * 3
                state_max = max(state_max, right_end)
                fig_max = max(fig_max, state_max)

            x = np.arange(state_min, state_max, float(state_max - state_min) / 10000.)
            y = np.zeros(len(x))
            for mixes in macros_data[word][state]:
                mean = macros_data[word][state][mixes]["mean"][feature_label]
                var = macros_data[word][state][mixes]["variance"][feature_label]
                weight = macros_data[word][state][mixes]["mixture_weight"]
                for count, x_in in enumerate(x):
                    y[count] += weight * gaussian(x_in, mean, var)
            gaussian_peaks.append(max(y))
            plt.plot(x, y, label="state {}, word {}".format(state, word), color=linecolors[color % 7], linestyle=linestyles[line % 4])
        
    plt.legend(loc="upper left")
    ax.set_title('Word(s): {}. Feature: {}'.format(words_str, feature_label))    
    ax.set_xlim([fig_min, fig_max])

    gaussian_peaks = np.array(gaussian_peaks)
    #ax.set_ylim([0, min(gaussian_peaks.mean() + 2*gaussian_peaks.std(), np.amax(gaussian_peaks) + 1)])

    fig.savefig(save_dir + '/{}:{}.png'.format(words_str.replace(', ','_'), feature_label))
    plt.close(fig)


def each_word_and_feature_graphs(feature_config_filepath, macros_filepath, save_dir, words, feature_labels):
    """Prebuilt function that generates a new graph for each pair combination (word, feature_label).
        Each graph consists of guassian plot(s) for each state

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that lists the features used  when testing and training the HMM

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM

    save_dir : str
        The directory where the graph plots are saved

    words: list of str
        A list of words which will each have its own graph
        If empty list, then all words are utilized

    feature_labels : list of str
        A list of features which will each have its own graph
        If empty list, then all features are utilized

    Returns
    -------
    None
        On success, a graph for each unique combination of word and feature_label is generated in the specified save directory.
        len(feature_labels) * len(words) graphs in total are generated

    """
    if not feature_labels:
        feature_labels = [ feature_label.rstrip() for feature_label in open(feature_config_filepath) ]

    if not words:
        words = get_macros(feature_config_filepath, macros_filepath).keys()

    for word in words:
        for feature_label in feature_labels:
            if 'landmark' in feature_label and int(feature_label.split('_')[-2]) >= 1: continue
            print("graph for word {} for feature {}".format(word, feature_label))
            generate_graph(feature_config_filepath, macros_filepath, save_dir, [word], feature_label)
        

def each_confused_word_and_feature_graphs(feature_config_filepath, macros_filepath, save_dir, feature_labels, confusion_matrix_filepath, threshold):
    """Prebuilt function that finds pair of confused words that exceeds the threshold from the confusion matrix. 
        Each pair of words for each feature_label generates a new graph. Each graph consists of guassian plot(s) for each state and both words

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that lists the features used  when testing and training the HMM

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM

    save_dir : str
        The directory where the graph plots are saved

    feature_labels : list of str
        A list of features which will each have its own graph
        If empty list, then all features are utilized
    
    confusion_matrix_filepath : str
        File path to the confusion matrix file that is generated from testing HMM

    threshold : float
        If the percentage of times a word is mislabeled exceeds the threshold, then the two words are confused (ground_truth, HMM classification)

    Returns
    -------
    None
        On success, a graph for each unique pair of confused words and feature_label is generated in the specified save directory.
        Depending on the threshold and confusion matrix, no graphs may be generated

    """
    if not feature_labels:
        feature_labels = [ feature_label.rstrip() for feature_label in open(feature_config_filepath) ]

    # confusion_matrix_dict: [ground_truth_word (vertical_axis_of_confusion_matrix)][predicted_word (horizontal_axis_of_confusion_matrix)]
    words = get_macros(feature_config_filepath, macros_filepath).keys()
    confusion_matrix_dict = get_confusion_matrix(confusion_matrix_filepath)['matrix']

    for row in confusion_matrix_dict.keys():
        total = sum(confusion_matrix_dict[row].values())
        if "Ins" in row or total == 0: continue

        for col in confusion_matrix_dict[row]:
            if not (row in col or col in row) and confusion_matrix_dict[row][col] / float(total) > float(threshold):
                row_word = None
                col_word = None
                for word in words:
                    if row in word: row_word = word
                    if col in word: col_word = word
                for feature_label in feature_labels:
                    print("graph for confused words ({}, {}) for feature {}".format(row_word, col_word, feature_label))
                    generate_graph(feature_config_filepath, macros_filepath, save_dir, [row_word, col_word], feature_label)

def plot_macros_gaussian(feature_config_filepath, macros_filepath, save_dir, words, feature_labels, confusion_matrix_filepath, threshold, mode):

    gaussian_dir = os.path.join(save_dir, 'visualization', 'gaussian', str(mode))
    if not os.path.exists(gaussian_dir):
        os.makedirs(gaussian_dir)

    if mode == 0:
        each_word_and_feature_graphs(feature_config_filepath, macros_filepath, gaussian_dir, words, feature_labels)
    elif mode == 1:
        each_confused_word_and_feature_graphs(feature_config_filepath, macros_filepath, gaussian_dir, feature_labels, confusion_matrix_filepath, threshold)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_config_filepath', default = '/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/configs/features.config')
    parser.add_argument('--macros_filepath', default = '/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/models/hmm45/newMacros')
    parser.add_argument('--save_dir', default = '/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/')
    parser.add_argument('--words', default = [])
    parser.add_argument('--feature_labels', default = [])
    parser.add_argument('--confusion_matrix_filepath', default = '/home/thad/copycat/copycat-ml/main/projects/prerna_20-03-25/hresults/res_hmm45.txt')
    parser.add_argument('--threshold', type = float, default = 0.05)
    parser.add_argument('--mode', type = int, default = 0)
    args = parser.parse_args()

    """Generates gaussian graphs for visualization purposes with respect to the newMacros data generated from HMMs.

    Parameters
    ----------
    feature_config_filepath : str
        File path to the feature config file that lists the features used  when testing and training the HMM

    macros_filepath : str
        File path to the corresponding newMacros result file that is generated from running HMM

    save_dir : str
    	The directory where the graph plots are saved

	words: list of str
		A list of words which will each have its own graph
		If empty list, then all words are utilized

   	feature_labels : list of str
   		A list of features which will each have its own graph
   		If empty list, then all features are utilized

   	confusion_matrix_filepath : str
   		File path to the confusion matrix file that is generated from testing HMM

   	threshold : float
   		If the percentage of times a word is mislabeled exceeds the threshold, then the two words are confused (ground_truth, HMM classification)

   	mode : int
   		If mode == 0, then plot each word and each feature seperately
   		If mode == 1, then plot each confused pair of words together for each feature

    """
    plot_macros_gaussian(args.feature_config_filepath, args.macros_filepath, args.save_dir, args.words, args.feature_labels, args.confusion_matrix_filepath, args.threshold, args.mode)