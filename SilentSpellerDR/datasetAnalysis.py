
import os
import sys
import glob
import argparse
from tqdm import tqdm
import shutil

import numpy as np
import pandas as pd

from createArkHtk import createMLF

def checkAndDelete(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)

def create(folder: str):
    os.mkdir(folder)

def getDataName(path: str):
    return path.split("/")[-1].split(".")[1]

def copyFile(source: str, destFolder: str):
    shutil.copy(source, destFolder) 

def getLabelFrequency(arkLocation: str):

    dataFiles = glob.glob(arkLocation)
    dataFiles = [getDataName(i) for i in dataFiles]
    dataFreq = {}
    dataLenFreq = {}
    dataLen = {}
    for dataFile in dataFiles:
        if dataFile in dataFreq:
            dataFreq[dataFile] += 1
        else:
            dataFreq[dataFile] = 1

        if len(dataFile) in dataLenFreq:
            dataLenFreq[len(dataFile)] += 1
        else:
            dataLenFreq[len(dataFile)] = 1

        if dataFile not in dataLen:
            dataLen[dataFile] = len(dataFile)

    return dataLenFreq

def trimDataSet(arkLocation: str, htkLocation: str, maxLength: int, maxNum: int):
    arkFolder = "ark"
    htkFolder = "htk"
    mlfFile = "all_labels.mlf"

    dataLenFreq = getLabelFrequency(arkLocation)
    # print("Data Length Frequency = " + str(dataLenFreq))
    checkAndDelete(arkFolder)
    checkAndDelete(htkFolder)
    create(arkFolder)
    create(htkFolder)

    arkFiles = glob.glob(arkLocation)
    htkFiles = glob.glob(htkLocation)
    arkFiles.sort()
    htkFiles.sort()

    total = 0

    for i in range(len(arkFiles)):
        if len(getDataName(arkFiles[i])) < maxLength and total < maxNum:
            copyFile(arkFiles[i], arkFolder)
            copyFile(htkFiles[i], htkFolder)
            total += 1
        
        # if len(getDataName(htkFiles[i])) < maxLength:
    
    createMLF(htkFolder+"/*", mlfFile)

if __name__ == "__main__":

    trimDataSet("/home/dhruva/Desktop/CopyCat/SilentSpeller/data/ark/*", "/home/dhruva/Desktop/CopyCat/SilentSpeller/data/htk/*", 10000000, 10000000000)