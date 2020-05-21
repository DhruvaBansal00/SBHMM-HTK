"""Defines method to train SBHMM

Methods
-------

trainSBHMM
"""
import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd

from .train import train
from src.test import test
from src.sbhmm import getClassifierFromStateAlignment
from src.prepare_data.ark_reader import read_ark_files
from src.prepare_data.ark_creation import _create_ark_file
from src.prepare_data.htk_creation import create_htk_files




def trainSBHMM(sbhmm_iters: int, train_iters: list, mean: float, variance: float, transition_prob: float, users: list, device: int) -> None:
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
    arkFileLoc = "data/ark/"

    for iters in range(sbhmm_iters):
        print("Training SBHMM")

        test(-2, -1, "alignment") #Save state alignments for each phrase in the results folder
        resultFile = glob.glob('results/*.mlf')[-1]

        trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFileLoc)

        """
        TODO: Create hmm and corresponding text files (basically prep data)
        Then you are ready to run another loop of training HMM models. 
        """

        
        arkFileSave = "data/arkSBHMM"+str(iters)+"/"
        htkFileSave = "data/htkSBHMM"+str(iters)
        if os.path.exists(arkFileSave):
            shutil.rmtree(arkFileSave)

        os.makedirs(arkFileSave)

        arkFiles = []
        if len(users) == 0:
            arkFiles = glob.glob(arkFileLoc+"*")
        else:
            for user in users:
                arkFiles.extend(glob.glob(arkFileLoc+user+"*"))

        print("Creating new arkFiles")
        for arkFile in arkFiles:

            content = read_ark_files(arkFile)
            newContent = trainedClassifier.getTransformedFeatures(content)
            #TODO: Perform PCA
            arkFileName = arkFile.split("/")[-1]
            arkFileSavePath = arkFileSave + arkFileName

            _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
        
        arkFileLoc = arkFileSave
        create_htk_files(htkFileSave, arkFileLoc + "*ark")







