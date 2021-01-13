"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 --users Naoki --hmm_insertion_penalty -70
    SBHMM EX = python3 driver.py --test_type standard --users Naoki --train_iters 25 50 --sbhmm_iters 25 50 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115
"""
"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 100 --users Prerna Linda | 
    HMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --users 02-22-20_Prerna_Android 04-29-20_Linda_Android 07-24-20_Matthew_4K --cross_val_method stratified --n_splits 10 --cv_parallel --parallel_jobs 10  --hmm_insertion_penalty -80
    SBHMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Prerna Linda --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115 --neighbors 70
    SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Linda Prerna --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -85 --sbhmm_insertion_penalty -85 --neighbors 70 --cross_val_method kfold --n_splits 10 --beam_threshold 2000.0
    SBHMM CV Parallel = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1  --include_word_level_states --include_word_position --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --neighbors 73 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 10
    Prepare Data = python3 driver.py --test_type none --prepare_data --users Matthew_4 Ishan_4 David_4
    Old SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 5 --multiple_classifiers
    SBHMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --sbhmm_iters 25 50 75 100 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --hmm_insertion_penalty 150 --sbhmm_insertion_penalty 150 --neighbors 200 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 7 --users Linda_4 Kanksha_4 Thad_4 Matthew_4 Prerna_4 David_4 Ishan_4
    HMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --hmm_insertion_penalty 10 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 4
"""
import sys
import glob
import argparse
import os
import shutil
import sys
import random
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)

sys.path.insert(0, '../../')
from src.prepare_data import prepare_data
from src.train import create_data_lists, train, trainSBHMM
from src.utils import get_results, save_results, load_json, get_arg_groups
from src.test import test, testSBHMM, verify
from joblib import Parallel, delayed
from statistics import mean

def returnUserDependentSplits(unique_users, htk_filepaths, test_size):
    splits = [[[],[]] for i in range(len(unique_users))]
    for htk_idx, curr_file in enumerate(htk_filepaths):
        curr_user = curr_file.split("/")[-1].split(".")[0].split('_')[-2]
        for usr_idx, usr in enumerate(unique_users):
            if usr == curr_user:
                if random.random() > test_size:
                    splits[usr_idx][0].append(htk_idx)
                else:
                    splits[usr_idx][1].append(htk_idx)
    splits = np.array(splits)
    return splits

def copyFiles(fileNames: list, newFolder: str, originalFolder: str, ext: str):
    if os.path.exists(newFolder):
        shutil.rmtree(newFolder)
    os.makedirs(newFolder)

    for currFile in fileNames:
        shutil.copyfile(os.path.join(originalFolder, currFile+ext), os.path.join(newFolder, currFile+ext))

def crossValFold(train_data: list, test_data: list, args: object, fold: int):
    print(f"Current split = {str(fold)}. Current Test data Size = {len(test_data)}")
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], args.phrase_len, fold)
    
    if args.train_sbhmm:
        classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold, os.path.join(str(fold), ""))
        testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                args.parallel_jobs, args.parallel_classifier_training, os.path.join(str(fold), ""))
    else:
        train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
        test(args.start, args.end, args.method, args.hmm_insertion_penalty, fold=os.path.join(str(fold), ""))

    if args.train_sbhmm:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.train_iters[-1]-1}.txt'    

    results = get_results(hresults_file)

    print(f'Current Word Error: {results["error"]}')
    print(f'Current Sentence Error: {results["sentence_error"]}')
    print(f'Current Insertion Error: {results["insertions"]}')
    print(f'Current Deletions Error: {results["deletions"]}')

    test(-1, -1, "alignment", args.hmm_insertion_penalty, beam_threshold=args.beam_threshold, fold=os.path.join(str(fold), ""))

    return [results['error'], results['sentence_error'], results['insertions'], results['deletions']]

    


def main():
    
    parser = argparse.ArgumentParser()
    ############################## ARGUMENTS #####################################
    #Important
    parser.add_argument('--prepare_data', action='store_true')
    parser.add_argument('--save_results', action='store_true')

    #Arguments for create_data_lists()
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['none', 'test_on_train', 'cross_val', 'standard'])
    parser.add_argument('--users', nargs='*', default=None)
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=24)
    parser.add_argument('--cross_val_method', required='cross_val' in sys.argv,
                        default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified',
                                                  'leave_one_user_out',
                                                  'user_dependent'
                                                  ])
    parser.add_argument('--n_splits', required='cross_val' in sys.argv,
                        type=int, default=10)
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.1)

    #Arguments for save_data()
    parser.add_argument('--save_results_file', type=str,
                        default='all_results.json')

    #Arguments for training
    parser.add_argument('--train_iters', nargs='*', type=int, default=[20, 50, 80])
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=0.00001)
    parser.add_argument('--transition_prob', type=float, default=0.6)
    parser.add_argument('--hmm_insertion_penalty', default=-10)

    #Arguments for SBHMM
    parser.add_argument('--train_sbhmm', action='store_true')    
    parser.add_argument('--sbhmm_cycles', type=int, default=1)
    parser.add_argument('--pca_components', type=int, default=32)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--lda_components', type=int, default=32)
    parser.add_argument('--lda', action='store_true')
    parser.add_argument('--sbhmm_iters', nargs='*', type=int, default=[20, 50, 80])
    parser.add_argument('--include_word_level_states', action='store_true')
    parser.add_argument('--include_word_position', action='store_true')
    parser.add_argument('--parallel_classifier_training', action='store_true')
    parser.add_argument('--parallel_jobs', default=4, type=int)
    parser.add_argument('--sbhmm_insertion_penalty', default=-10)
    parser.add_argument('--neighbors', default=50)
    parser.add_argument('--multiple_classifiers', action='store_true')
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn', 'adaboost'])
    parser.add_argument('--beam_threshold', default=100000.0)

    #Arguments for testing
    parser.add_argument('--start', type=int, default=-2)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--method', default='recognition', 
                        choices=['recognition', 'verification'])
    parser.add_argument('--acceptance_threshold', default=-150)
    
    args = parser.parse_args()
    ########################################################################################

    cross_val_methods = {'kfold': (KFold, True),
                         'leave_one_phrase_out': (LeaveOneGroupOut(), True),
                         'stratified': (StratifiedKFold, True),
                         'leave_one_user_out': (LeaveOneGroupOut(), True),
                         'user_dependent': (None, False)
                         }
    cvm = args.cross_val_method
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    features_config = load_json('configs/features.json')
    all_results = {'features': features_config['selected_features'],
                   'average': {}}
                   
    if args.train_sbhmm:
        hresults_file = f'hresults/res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/res_hmm{args.train_iters[-1]-1}.txt'

    if args.prepare_data:
        
        prepare_data(features_config, args.users)

    if args.test_type == 'none':
        sys.exit()

    elif args.test_type == 'test_on_train':
        
        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        create_data_lists(htk_filepaths, htk_filepaths, args.phrase_len)
        
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty,
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob)
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            elif args.method == "verification":
                correct, incorrect = verify(args.end, args.insertion_penalty, args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['fold_0']['correct'] = correct
            all_results['fold_0']['incorrect'] = incorrect

            print('Test on Train Results')
    
    elif args.test_type == 'cross_val' and args.cv_parallel:
        print("You have invoked parallel cross validation. Be prepared for dancing progress bars!")

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [filepath.split('/')[-1].split(".")[0] + " " + ' '.join(filepath.split('/')[-1].split(".")[1].split("_"))
            for filepath
            in htk_filepaths]
        

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            print(len(unique_phrases), len(phrases))
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]      
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            users = [filepath.split('/')[-1].split('.')[0].split('_')[-2]
                for filepath
                in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]            
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [filepath.split('/')[-1].split('.')[0].split('_')[-2]
                for filepath
                in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))        
        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)(np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold) for currFold in range(len(splits)))
        
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])
        print(stats)

    elif args.test_type == 'cross_val':


        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]
        
        users = [filepath.split('/')[-1].split('.')[0].split('_')[1]
            for filepath
            in htk_filepaths]     

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]      
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            unique_users = set(users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]            
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [filepath.split('/')[-1].split('.')[0].split('_')[-2]
                for filepath
                in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))

        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_data = np.array(htk_filepaths)[train_index]
            test_data = np.array(htk_filepaths)[test_index]

            phrase = np.array(phrases)[test_index][0]
            phrase_len = len(phrase.split(' '))
            phrase_count = len(test_data)
            word_count = phrase_len * phrase_count
            word_counts.append(word_count)
            phrase_counts.append(phrase_count)
            create_data_lists(train_data, test_data, args.phrase_len)

            if args.train_sbhmm:
                classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
                testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                        args.parallel_jobs, args.parallel_classifier_training)
            else:
                train(args.train_iters, args.mean, args.variance, args.transition_prob)
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            
            results = get_results(hresults_file)
            all_results[f'fold_{i}'] = results
            all_results[f'fold_{i}']['phrase'] = phrase
            all_results[f'fold_{i}']['phrase_count'] = phrase_count

            print(f'Current Word Error: {results["error"]}')
            print(f'Current Sentence Error: {results["sentence_error"]}')

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

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        phrases = [' '.join(filepath.split('.')[1].split('_'))
            for filepath
            in htk_filepaths]
        #unique_phrases = set(phrases)
        #group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
        #groups = [group_map[phrase] for phrase in phrases]
        train_data, test_data, _, _ = train_test_split(
            htk_filepaths, phrases, test_size=args.test_size,
            random_state=args.random_state)

        create_data_lists(train_data, test_data, args.phrase_len)
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob)
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            elif args.method == "verification":
                correct, incorrect = verify(args.end, args.insertion_penalty, args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['average']['correct'] = correct
            all_results['average']['incorrect'] = incorrect

            print('Standard Train/Test Split Results')

    if args.method == "recognition":
        
        print(f'Average Error: {all_results["average"]["error"]}')
        print(f'Average Sentence Error: {all_results["average"]["sentence_error"]}')
        
        if args.test_type == 'cross_val' and args.cv_parallel:
            print(f'Average Insertions: {all_results["average"]["insertions"]}')
            print(f'Average Deletions: {all_results["average"]["deletions"]}')
    
    if args.method == "verification":

        print(f'Videos correct: {all_results["average"]["correct"]}')
        print(f'Videos incorrect: {all_results["average"]["incorrect"]}')
        percent_correct = all_results["average"]["correct"]/(all_results["average"]["correct"] + all_results["average"]["incorrect"])
        print(f'Correct %: {percent_correct}')

    # print(all_results)
    # Loads data as new run into pickle
    if args.save_results:
        save_results(all_results, args.save_results_file, 'a')