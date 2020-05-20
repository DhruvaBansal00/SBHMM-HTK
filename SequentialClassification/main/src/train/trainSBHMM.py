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

from .train import train, initialize_models, generate_prototype
from src.test import test
from src.sbhmm import getClassifierFromStateAlignment
from src.utils import load_json



def trainSBHMM(sbhmm_iters: int, train_iters: list, mean: float, variance: float, transition_prob: float, device: int) -> None:
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

        test(-2, -1, "alignment") #Save state alignments for each phrase in the results folder
        resultFile = glob.glob('results/*.mlf')[-1]

        trainedClassifier = getClassifierFromStateAlignment(resultFile)


