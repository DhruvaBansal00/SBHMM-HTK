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
from tqdm import tqdm
from sklearn.decomposition import PCA

from .train import train
from src.test import test
from src.sbhmm import getClassifierFromStateAlignment
from src.prepare_data.ark_reader import read_ark_files
from src.prepare_data.ark_creation import _create_ark_file
from src.prepare_data.htk_creation import create_htk_files





def trainSBHMM(sbhmm_cycles: int, train_iters: list, mean: float, variance: float, 
            transition_prob: float, device: int, pca_components: int, sbhmm_iters: list, 
            include_state: bool, include_index: bool, no_pca: bool, insertion_penalty: int,
            n_jobs: int, parallel: bool) -> None:
    """Trains the SBHMM using HTK. First completes a loop of
    training HMM as usual. Then completes as many iterations of 
    adaboosting + HMM training as specified.

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """
    print("----------------Starting SBHMM training with basic HMM for alignment-------------------")
    train(train_iters, mean, variance, transition_prob, device)
    arkFileLoc = "data/ark/"
    htkFileLoc = "data/htk/"
    trainDataFile = "lists/train.data"
    
    classifiers = []

    for iters in range(sbhmm_cycles):

        test(-2, -1, "alignment", insertion_penalty) #Save state alignments for each phrase in the results folder
        resultFile = glob.glob('results/*.mlf')[-1]

        trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFileLoc, include_state=include_state, 
                        include_index=include_index, n_jobs=n_jobs, parallel=parallel)
        classifiers.append(trainedClassifier)
        
        arkFileSave = "data/arkSBHMM"+str(iters)+"/"
        htkFileSave = "data/htkSBHMM"+str(iters)+"/"
        if os.path.exists(arkFileSave):
            shutil.rmtree(arkFileSave)

        os.makedirs(arkFileSave)

        arkFiles = []
        newHtkFiles = []
        with open(trainDataFile, 'r') as trainData:
            for path in trainData:
                arkFiles.append(path.replace(htkFileLoc, arkFileLoc).replace(".htk", ".ark").strip('\n'))
                newHtkFiles.append(path.replace(htkFileLoc, htkFileSave))

        print("Creating new .ark Files")
        num_features = 0
        for arkFile in tqdm(arkFiles):

            content = read_ark_files(arkFile)
            newContent = trainedClassifier.getTransformedFeatures(content)
            
            if not no_pca:
                pca = PCA(n_components=pca_components)
                newContent = pca.fit_transform(newContent)

            num_features = newContent.shape[1]
            arkFileName = arkFile.split("/")[-1]
            arkFileSavePath = arkFileSave + arkFileName

            _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
        
        print("Creating new .htk Files")
        create_htk_files(htkFileSave, arkFileSave + "*ark")

        arkFileLoc = arkFileSave
        htkFileLoc = htkFileSave

        print("Re-writing lists/train.data")
        with open(trainDataFile, 'w') as trainData:
            trainData.writelines(newHtkFiles)
            trainData.close()

        print("Training HMM on new feature space")
        train(sbhmm_iters, mean, variance, transition_prob, device, num_features=num_features)

    return classifiers






