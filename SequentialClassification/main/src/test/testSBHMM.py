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

from .test import test
from src.prepare_data.ark_reader import read_ark_files
from src.prepare_data.ark_creation import _create_ark_file
from src.prepare_data.htk_creation import create_htk_files

def testSBHMM(start: int, end: int, method: str, classifiers: []) -> None:

    print("-------Testing SBHMM-----------")
    testDataFile = "lists/test.data"
    htkFileLoc = "data/htk/"
    arkFileLoc = "data/ark/"

    arkFileSave = "data/arkSBHMMTest/"
    htkFileSave = "data/htkSBHMMTest/"

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
            newContent = classifier.getTransformedFeatures(newContent)
        
        arkFileName = arkFile.split("/")[-1]
        arkFileSavePath = arkFileSave + arkFileName
        _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
    
    print("------Creating new HTK files--------")
    create_htk_files(htkFileSave, arkFileSave + "*ark")
    with open(testDataFile, 'w') as testData:
        testData.writelines(newHtkFiles)
        testData.close()

    print("------Executing HVITE Command-------")
    test(start, end, method)

        



