"""Defines method to verify phases using HMM.

Methods
-------
verify
"""
import os
import glob
import shutil
import numpy as np
from string import Template
import tqdm

def return_average_ll(file_path: str):
    total = 0
    num = 0
    with open(file_path) as verification_path:
        verification_path.readline()
        verification_path.readline()
        for line in verification_path:
            numbers = line.split(" ")
            if len(numbers) > 1:
                total += float(numbers[3])
                num += 1
                if len(numbers) > 4:
                    total += float(numbers[5])
                    num += 1
    
    return total/num

def verification_cmd(model_iter: int, insertion_penalty: int, verification_list: str, label_file: str, 
                    beam_threshold: int = 2000, fold: str = ""):

    HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
                     f'{verification_list} -i $results -t {beam_threshold} '
                     f'-p {insertion_penalty} -I {label_file} -s 25 dict wordList '
                     f'>/dev/null 2>&1')
    HVite_cmd = Template(HVite_str)
    macros_filepath = f'models/{fold}hmm{model_iter}/newMacros'
    results_filepath = f'results/{fold}res_hmm{model_iter}.mlf'

    os.system(HVite_cmd.substitute(macros=macros_filepath, results=results_filepath))

def return_average_ll_per_sign(model_iter:int, insertion_penalty: int, acceptance_threshold: int, 
                            beam_threshold: int = 2000, fold: str = "") -> None:
    
    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.

    average_ll_per_sign = {}

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")

            with open(curr_verification_phrase, "w") as verification_list:
                verification_list.write(curr_video_path+"\n")
            with open(curr_verification_label, "w") as verification_label:
                verification_label.write("#!MLF!#\n")
                verification_label.write(label_file_path)
                verification_label.write("sil0\n")
                for word in correct_phrase.split("_"):
                    verification_label.write(word+"\n")
                verification_label.write("sil1\n")
                verification_label.write(".\n")
                
            verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                            curr_verification_label, beam_threshold, fold)
            curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf')
            
            if correct_phrase in average_ll_per_sign:
                average_ll_per_sign[correct_phrase].append(curr_average)
            else:
                average_ll_per_sign[correct_phrase] = [curr_average]

    for phrase in average_ll_per_sign:
        average_ll_per_sign[phrase] = np.mean(np.array(average_ll_per_sign[phrase]))
    
    return average_ll_per_sign


'''
    While evaluating network accuracy:
        For each video, calculate how many times it correctly verifies it. Also calculate how many times
        it incorrectly verifies correctly rejects other phrases. Report #correct_labels/total_labels.

        For calculating log likelihood probability, make a list of the phrase you want to check and 
        an mlf file with the label corresponding to that phase. Change this label to perform alignment
        with other phrases.

    For now, use a threshold on the average log likelihood probability for verifying or rejecting.
'''
def verify_simple(model_iter:int, insertion_penalty: int, acceptance_threshold: int, 
                beam_threshold: int = 2000, fold: str = "") -> None:

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    train_phrases = f'lists/{fold}train.data'
    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.
    unique_phrases = set()

    with open(train_phrases) as file:
        for line in file:
            curr_phrase = line.split("/")[-1].split(".")[1]
            unique_phrases.add(curr_phrase)
    
    positive = 0
    false_positive = 0
    false_negative = 0
    negative = 0

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")

            for curr_phrase in unique_phrases:
                with open(curr_verification_phrase, "w") as verification_list:
                    verification_list.write(curr_video_path+"\n")
                with open(curr_verification_label, "w") as verification_label:
                    verification_label.write("#!MLF!#\n")
                    verification_label.write(label_file_path)
                    verification_label.write("sil0\n")
                    for word in curr_phrase.split("_"):
                        verification_label.write(word+"\n")
                    verification_label.write("sil1\n")
                    verification_label.write(".\n")
                
                verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                                curr_verification_label, beam_threshold, fold)
                curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf')

                if (correct_phrase == curr_phrase and curr_average >= acceptance_threshold):
                    positive += 1
                elif (correct_phrase != curr_phrase and curr_average < acceptance_threshold):
                    negative += 1
                elif (correct_phrase != curr_phrase and curr_average >= acceptance_threshold) :
                    false_positive += 1
                else :
                    false_negative += 1

    return positive, negative, false_positive, false_negative 

