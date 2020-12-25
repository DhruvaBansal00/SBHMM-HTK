"""Defines method to test SBHMMHMM. Can perform verification or recognition.

Methods
-------
test
"""
import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

from .test import test
from src.prepare_data.ark_reader import read_ark_files
from src.prepare_data.ark_creation import _create_ark_file
from src.prepare_data.htk_creation import create_htk_files

def testSBHMM(start: int, end: int, method: str, classifiers: [], pca_components: int, pca: bool,
            sbhmm_interstion_penalty: float, n_jobs: int, parallel: bool, fold: str = "") -> None:

    print("-------Testing SBHMM-----------")
    testDataFile = f"lists/{fold}test.data"
    htkFileLoc = f"data/{fold}htk/"
    arkFileLoc = f"data/{fold}ark/"

    arkFileSave = f"data/{fold}arkSBHMMTest/"
    htkFileSave = f"data/{fold}htkSBHMMTest/"

    if os.path.exists(arkFileSave):
            shutil.rmtree(arkFileSave)
    os.makedirs(arkFileSave)
    if os.path.exists(htkFileSave):
            shutil.rmtree(htkFileSave)
    os.makedirs(htkFileSave)

    arkFiles = []
    newHtkFiles = []
    with open(testDataFile, 'r') as testData:
        for path in testData:
            arkFiles.append(path.replace(htkFileLoc, arkFileLoc).replace(".htk", ".ark").strip('\n'))
            newHtkFiles.append(path.replace(htkFileLoc, htkFileSave))
    
    print("-----Creating new ARK files--------")
    for arkFile in tqdm(arkFiles):

        content = read_ark_files(arkFile)
        newContent = content
        for classifier in classifiers:
            newContent = classifier.getTransformedFeatures(newContent, parallel, n_jobs)
            if pca:
                pca_model = PCA(n_components=pca_components)
                newContent = pca_model.fit_transform(newContent)

        arkFileName = arkFile.split("/")[-1]
        arkFileSavePath = arkFileSave + arkFileName
        _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
    
    print("------Creating new HTK files--------")
    create_htk_files(htkFileSave, arkFileSave + "*ark")
    with open(testDataFile, 'w') as testData:
        testData.writelines(newHtkFiles)
        testData.close()

    print("------Executing HVITE Command-------")
    test(start, end, method, sbhmm_interstion_penalty, fold=fold)

        



