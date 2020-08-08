"""Defines method to train HMM and parser group to pass arguments to
train method.

Methods
-------
train_cli
train
"""
import os
import sys
import glob
import shutil
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from .gen_init_models_each_word import initialize_models
from .gen_prototype import generate_prototype
from src.utils import load_json


def train(train_iters: list, mean: float, variance: float, transition_prob: float, num_features: int = None, fold: str = "") -> None:
    """Trains the HMM using HTK. Calls HCompV, HRest, HERest, HHEd, and
    HParse. Configuration files for prototypes and increasing mixtures
    are found in configs/. 

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    if os.path.exists(f'models/{fold}'):
        shutil.rmtree(f'models/{fold}')

    if os.path.exists(f'logs/{fold}'):
        if os.path.exists(f'logs/{fold}train.log'):
            os.remove(f'logs/{fold}train.log')

    os.makedirs(f'models/{fold}')

    if not os.path.exists(f'logs/{fold}'):
        os.makedirs(f'logs/{fold}')

    #n_models = train_iters[-1] + len(train_iters) - 1
    for i in range(train_iters[-1] + 1):
        hmm_dir = os.path.join('models', f'{fold}hmm{i}')
        if not os.path.exists(hmm_dir):
            os.makedirs(hmm_dir)

    features_config = load_json('configs/features.json')
    
    if num_features is None:
        n_features = len(features_config['selected_features'])
    else:
        n_features = num_features

    print("-------------- Training HMM --------------")

    prototypes_config = load_json('configs/prototypes.json')
    for n_states in prototypes_config:

        prototype_filepath = f'models/{fold}prototype'
        generate_prototype(
            int(n_states), n_features, prototype_filepath, mean,
            variance, transition_prob)    

        print('Running HCompV...')
        HCompV_command = (f'HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 '
                          f'-m -S lists/{fold}train.data -M models/{fold}hmm0 '
                          f'{prototype_filepath} >> logs/{fold}train.log')
        os.system(HCompV_command)
        print('HCompV Complete')

        initialize_models(f'models/{fold}hmm0/prototype', prototypes_config[n_states], f'models/{fold}hmm0')
        #initialize_models('models/prototype', 'wordList', 'models/hmm0')

    hmm0_files = set(glob.glob(f'models/{fold}hmm0/*')) - {f'models/{fold}hmm0/vFloors'}
    for hmm0_file in tqdm(hmm0_files):

        # print(f'Running HRest for {hmm0_file}...')
        HRest_command = (f'HRest -A -i 60 -C configs/hrest.conf -v 0.1 -I '
                         f'all_labels.mlf -M models/{fold}hmm1 -S lists/{fold}train.data '
                         f'{hmm0_file} >> logs/{fold}train.log')
        os.system(HRest_command)
    print('HRest Complete')


    print('Running HERest Iteration: 1...')
    HERest_command = (f'HERest -A -d models/{fold}hmm1 -c 500.0 -v 0.0005 -I '
                      f'all_labels.mlf -M models/{fold}hmm2 -S lists/{fold}train.data -T '
                      f'1 wordList >> logs/{fold}train.log')
    os.system(HERest_command)

    start = 2
    for i, n_iters in enumerate(train_iters):

        for iter_ in tqdm(range(start, n_iters)):

            # print(f'Running HERest Iteration: {iter_}...')
            HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -A -H '
                            f'models/{fold}hmm{iter_}/newMacros -I all_labels.mlf -M '
                            f'models/{fold}hmm{iter_+1} -S lists/{fold}train.data -T 1 wordList '
                            f'>> logs/{fold}train.log')
            os.system(HERest_command)
        print('HERest Complete')

        if n_iters != train_iters[-1]:
            print(f'Running HHed Iteration: {n_iters}...')
            HHed_command = (f'HHEd -A -H models/{fold}hmm{n_iters-1}/newMacros -M '
                            f'models/{fold}hmm{n_iters} configs/hhed{i}.conf '
                            f'wordList')
            os.system(HHed_command)
            print('HHed Complete')
            start = n_iters

    cmd = 'HParse -A -T 1 grammar.txt wordNet.txt'
    os.system(cmd)
