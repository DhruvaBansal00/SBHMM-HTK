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

sys.path.insert(0, '../../')
from src.train import initialize_models, generate_prototype
from src.utils import load_json


def train_cli(parser: ArgumentParser) -> ArgumentParser:
    """Defines arguments to be passed to train() method.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to which to add the argument group for training.

    Returns
    -------
    return_val : ArgumentParser
        The parser with the training argument group added.
    """

    group = parser.add_argument_group('train_args')
    group.add_argument('--train_iters', nargs='*', type=int, default=[20, 45, 60, 80])
    group.add_argument('--mean', type=float, default=0.0)
    group.add_argument('--variance', type=float, default=1.0)
    group.add_argument('--transition_prob', type=float, default=0.6)
    # parser.add_argument('--device', type=int, default=0) # 0 is for mediapipe, 1 is for kinect

    return parser


def train(train_args: Namespace, device) -> None:
    """Trains the HMM using HTK. Calls HCompV, HRest, HERest, HHEd, and
    HParse. Configuration files for prototypes and increasing mixtures
    are found in configs/. 

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    if os.path.exists('models'):
        shutil.rmtree('models')

    if os.path.exists('logs'):
        if os.path.exists('logs/train.log'):
            os.remove('logs/train.log')

    os.makedirs('models')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    #n_models = train_iters[-1] + len(train_iters) - 1
    for i in range(train_args.train_iters[-1] + 1):
        hmm_dir = os.path.join('models', f'hmm{i}')
        if not os.path.exists(hmm_dir):
            os.makedirs(hmm_dir)
    
    if(device==0):
        features_config = load_json('configs/features.json')
    else:
        features_config = load_json('configs/features_kinect.json')
    n_features = len(features_config['selected_features'])

    print("-------------- Training HMM --------------")

    prototypes_config = load_json('configs/prototypes.json')
    for n_states in prototypes_config:

        prototype_filepath = 'models/prototype'
        generate_prototype(
            int(n_states), n_features, prototype_filepath, train_args.mean,
            train_args.variance, train_args.transition_prob)    

        print('Running HCompV...')
        HCompV_command = (f'HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 '
                          f'-m -S lists/train.data -M models/hmm0 '
                          f'{prototype_filepath} >> logs/train.log')
        os.system(HCompV_command)
        print('HCompV Complete')

        initialize_models(f'models/hmm0/prototype', prototypes_config[n_states], 'models/hmm0')
        #initialize_models('models/prototype', 'wordList', 'models/hmm0')

    hmm0_files = set(glob.glob('models/hmm0/*')) - {'models/hmm0/vFloors'}
    for hmm0_file in hmm0_files:

        print(f'Running HRest for {hmm0_file}...')
        HRest_command = (f'HRest -A -i 60 -C configs/hrest.conf -v 0.1 -I '
                         f'all_labels.mlf -M models/hmm1 -S lists/train.data '
                         f'{hmm0_file} >> logs/train.log')
        os.system(HRest_command)
    print('HRest Complete')


    print('Running HERest Iteration: 1...')
    HERest_command = (f'HERest -A -d models/hmm1 -c 500.0 -v 0.0005 -A -I '
                      f'all_labels.mlf -M models/hmm2 -S lists/train.data -T '
                      f'1 wordList >> logs/train.log')
    os.system(HERest_command)

    start = 2
    for i, n_iters in enumerate(train_args.train_iters):

        for iter_ in range(start, n_iters):

            print(f'Running HERest Iteration: {iter_}...')
            HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -A -H '
                            f'models/hmm{iter_}/newMacros -I all_labels.mlf -M '
                            f'models/hmm{iter_+1} -S lists/train.data -T 1 wordList '
                            f'>> logs/train.log')
            os.system(HERest_command)
        print('HERest Complete')

        if n_iters != train_args.train_iters[-1]:
            print(f'Running HHed Iteration: {n_iters}...')
            HHed_command = (f'HHEd -A -H models/hmm{n_iters-1}/newMacros -M '
                            f'models/hmm{n_iters} configs/hhed{i}.conf '
                            f'wordList')
            os.system(HHed_command)
            print('HHed Complete')
            start = n_iters

    cmd = 'HParse -A -T 1 grammar.txt wordNet.txt'
    os.system(cmd)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = train_cli(parser)
    args = parser.parse_args()

    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    test_errors = []
    htk_filepaths = glob.glob('data/htk/*htk')

    for train_index, test_index in kf.split(htk_filepaths):

        print("TRAIN:", train_index, "TEST:", test_index)
        train, test = htk_filepaths[train_index], htk_filepaths[test_index]

        create_data_lists(train, test)

        train(args.initial_iters, args.reestimate_iters, args.users,
              args.phrase_len, args.n_states, args.mean, args.variance,
              args.transition_prob)

        test_error = get_test_error()

        test_errors.append()
