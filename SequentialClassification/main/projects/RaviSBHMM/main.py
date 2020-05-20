"""Main file used to prepare training data, train, and test HMMs.
"""
import sys
import glob
import argparse

import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)

sys.path.insert(0, '../../')
from src.prepare_data import prepare_data
from src.train import create_data_lists, train, trainSBHMM
from src.utils import get_results, save_results, load_json, get_arg_groups
from src.test import test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ############################## ARGUMENTS #####################################
    #Important
    parser.add_argument('--prepare_data', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--device', type=int, default=0) # 0 is for mediapipe, 1 is for kinect

    #Arguments for create_data_lists()
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['test_on_train', 'cross_val', 'standard'])
    parser.add_argument('--users', nargs='*', default=[])
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=24)
    parser.add_argument('--cross_val_method', required='cross_val' in sys.argv,
                        default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified'])
    parser.add_argument('--n_splits', required='cross_val' in sys.argv,
                        type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)

    #Arguments for save_data()
    parser.add_argument('--save_results_file', type=str,
                        default='all_results.json')

    #Arguments for training
    parser.add_argument('--train_sbhmm', action='store_true')    
    parser.add_argument('--train_iters', nargs='*', type=int, default=[15, 20])
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=1.0)
    parser.add_argument('--transition_prob', type=float, default=0.6)
    parser.add_argument('--sbhmm_iters', type=int, default=1)

    #Arguments for testing
    parser.add_argument('--start', type=int, default=-10)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--method', default='recognition')
    
    args = parser.parse_args()
    ########################################################################################

    cross_val_methods = {'kfold': (KFold, False),
                         'leave_one_phrase_out': (LeaveOneGroupOut, True),
                         'stratified': (StratifiedKFold, True)}
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    features_config = load_json('configs/features.json')
    all_results = {'features': features_config['selected_features'],
                   'average': {}}
    hresults_file = f'hresults/res_hmm{args.train_iters[-1]-1}.txt'

    if args.prepare_data:

        prepare_data(features_config, args.device)

    if args.test_type == 'test_on_train':

        htk_filepaths = glob.glob('data/htk/*htk')

        create_data_lists(
            htk_filepaths, [], args.users, args.phrase_len, test_on_train=True)
        
        if args.train_sbhmm:
            trainSBHMM(args.sbhmm_iters, args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
        test(args.start, args.end, args.method)

        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')

    elif args.test_type == 'cross_val':

        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0
        htk_filepaths = glob.glob('data/htk/*htk')

        if(args.device==0):
            phrases = [' '.join(filepath.split('_')[1:-1])
               for filepath
               in htk_filepaths]
        else:
            # uncomment for prerna
            # phrases = [' '.join(filepath.split('/')[-1].split('.')[0].split('_')[0:-1]) 
            #     for filepath 
            #     in htk_filepaths]
            # for linda
            phrases = []
            for filepath in htk_filepaths:
                if('error' in ' '.join(filepath.split('/')[-1].split('.')[0])):
                    phrases.append(' '.join(filepath.split('/')[-1].split('.')[0].split('_')[0:-3]))
                else:
                    phrases.append(' '.join(filepath.split('/')[-1].split('.')[0].split('_')[0:-2]))
        
        unique_phrases = set(phrases)
        group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
        groups = [group_map[phrase] for phrase in phrases]
        cross_val = cross_val_method(n_splits=args.n_splits)

        if use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))

        for i, (train_index, test_index) in enumerate(splits):

            train_data = np.array(htk_filepaths)[train_index]
            test_data = np.array(htk_filepaths)[test_index]

            phrase = np.array(phrases)[test_index][0]
            phrase_len = len(phrase.split(' '))
            phrase_count = len(test_data)
            word_count = phrase_len * phrase_count
            word_counts.append(word_count)
            phrase_counts.append(phrase_count)
            create_data_lists(
                train_data, test_data, args.users, args.phrase_len)
            if args.train_sbhmm:
                trainSBHMM(args.sbhmm_iters, args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
            else:
                train(args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
            test(args.start, args.end, args.method)

            results = get_results(hresults_file)
            all_results[f'fold_{i}'] = results
            all_results[f'fold_{i}']['phrase'] = phrase
            all_results[f'fold_{i}']['phrase_count'] = phrase_count

            substitutions += (word_count * results['substitutions'] / 100)
            deletions += (word_count * results['deletions'] / 100)
            insertions += (word_count * results['insertions'] / 100)
            sentence_errors += (phrase_count * results['sentence_error'] / 100)

        total_words = sum(word_counts)
        total_phrases = sum(phrase_counts)
        total_errors = substitutions + deletions + insertions
        mean_error = (total_errors / total_words) * 100
        mean_error = np.round(mean_error, 4)
        mean_sentence_error = (sentence_errors / total_phrases) * 100
        mean_sentence_error = np.round(mean_sentence_error, 2)

        all_results['average']['error'] = mean_error
        all_results['average']['sentence_error'] = mean_sentence_error

        print('Cross-Validation Results')

    elif args.test_type == 'standard':

        htk_filepaths = glob.glob('data/htk/*htk')
        if(args.device==0):
            phrases = [' '.join(filepath.split('_')[1:-1])
               for filepath
               in htk_filepaths]
        else:
            phrases = [' '.join(filepath.split('/')[-1].split('.')[0].split('_')[0:-1]) 
                for filepath 
                in htk_filepaths]
        #unique_phrases = set(phrases)
        #group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
        #groups = [group_map[phrase] for phrase in phrases]
        train_data, test_data, _, _ = train_test_split(
            htk_filepaths, phrases, test_size=args.test_size,
            random_state=args.random_state)

        create_data_lists(train_data, test_data, args.users, args.phrase_len)
        if args.train_sbhmm:
            trainSBHMM(args.sbhmm_iters, args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob, args.device)
        test(args.start, args.end, args.method)

        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['correct'] = all_results['fold_0']['correct']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Standard Train/Test Split Results')

    if args.method == "recognition":
        
        print(f'Average Error: {all_results["average"]["error"]}')
        print(f'Average Sentence Error: {all_results["average"]["sentence_error"]}')

    # print(all_results)
    # Loads data as new run into pickle
    if args.save_results:
        save_results(all_results, args.save_results_file)