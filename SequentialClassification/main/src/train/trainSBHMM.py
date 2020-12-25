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
from joblib import Parallel, delayed

from .train import train
from src.test import test
from src.sbhmm import getClassifierFromStateAlignment
from src.prepare_data.ark_reader import read_ark_files
from src.prepare_data.ark_creation import _create_ark_file
from src.prepare_data.htk_creation import create_htk_files


def createNewArkFile(arkFile: str, trainedClassifier: object, pca_components: int, pca: bool, 
                    arkFileSave: str, parallel: bool, n_jobs: int):
    content = read_ark_files(arkFile)
    newContent = trainedClassifier.getTransformedFeatures(content, parallel, n_jobs)
    
    if pca:
        pca_model = PCA(n_components=pca_components)
        newContent = pca_model.fit_transform(newContent)
        newContent *= 1000

    num_features = newContent.shape[1]
    arkFileName = arkFile.split("/")[-1]
    arkFileSavePath = arkFileSave + arkFileName

    _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
    return num_features



def trainSBHMM(sbhmm_cycles: int, train_iters: list, mean: float, variance: float, 
            transition_prob: float, pca_components: int, sbhmm_iters: list, 
            include_state: bool, include_index: bool, pca: bool, hmm_insertion_penalty: float, sbhmm_insertion_penalty: float,
            n_jobs: int, parallel: bool, trainMultipleClassifiers: bool, knn_neighbors: str, classifier: str,
            beam_threshold: float, fold: str = "") -> object:
    """Trains the SBHMM using HTK. First completes a loop of
    training HMM as usual. Then completes as many iterations of 
    KNN + HMM training as specified.

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """
    print("----------------Starting SBHMM training with basic HMM for alignment-------------------")
    train(train_iters, mean, variance, transition_prob, fold=fold)
    arkFileLoc = f"data/{fold}ark/"
    htkFileLoc = f"data/{fold}htk/"
    trainDataFile = f"lists/{fold}train.data"
    
    classifiers = []

    for iters in range(sbhmm_cycles):
        test(-2, -1, "alignment", hmm_insertion_penalty if iters == 0 else sbhmm_insertion_penalty, beam_threshold=beam_threshold, fold=fold) #Save state alignments for each phrase in the results folder
        resultFile = glob.glob(f'results/{fold}*.mlf')[-1]

        trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFileLoc, classifier=classifier, include_state=include_state, 
                        include_index=include_index, n_jobs=n_jobs, parallel=parallel, trainMultipleClassifiers=trainMultipleClassifiers,
                        knn_neighbors=int(knn_neighbors))
        classifiers.append(trainedClassifier)
        
        arkFileSave = f"data/{fold}arkSBHMM"+str(iters)+"/"
        htkFileSave = f"data/{fold}htkSBHMM"+str(iters)+"/"
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
            num_features = createNewArkFile(arkFile, trainedClassifier, pca_components, pca, arkFileSave, parallel, n_jobs)
        
        print("Creating new .htk Files")
        create_htk_files(htkFileSave, arkFileSave + "*ark")

        arkFileLoc = arkFileSave
        htkFileLoc = htkFileSave

        print(f"Re-writing lists/{fold}train.data")
        with open(trainDataFile, 'w') as trainData:
            trainData.writelines(newHtkFiles)
            trainData.close()

        print("Training HMM on new feature space")
        train(sbhmm_iters, mean, variance, transition_prob, num_features=num_features, fold=fold)

    return classifiers






