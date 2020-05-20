"""Defines method to train SBHMM

Methods
-------

trainSBHMM
"""
import os
import sys
import glob
import shutil
from argparse import ArgumentParser, Namespace
sys.path.insert(0, '../../')
from train import train
from src.train import initialize_models, generate_prototype
from src.utils import load_json


def trainSBHMM(sbhmm_iters, train_iters, mean, variance, transition_prob, device) -> None:
    """Trains the SBHMM using HTK. First completes a loop of
    training HMM as usual. Then completes as many iterations of 
    adaboosting + HMM training as specified.

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    train(train_iters, mean, variance, transition_prob, device)
    for iters in range(sbhmm_iters):
        print("Training SBHMM")
