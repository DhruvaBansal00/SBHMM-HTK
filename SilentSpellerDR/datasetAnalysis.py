
import os
import sys
import glob
import argparse
from tqdm import tqdm
import shutil

import numpy as np
import pandas as pd

def getLabelFrequency(arkLocation: str):

    dataFiles = glob.glob(arkLocation)
    dataFiles = [i.split("/")[-1].split(".")[1] for i in dataFiles]
    dataFreq = {}
    for dataFile in dataFiles:
        if dataFile in dataFreq:
            dataFreq[dataFile] += 1
        else:
            dataFreq[dataFile] = 1
    
    print(dataFreq)

getLabelFrequency("/home/dhruva/Desktop/CopyCat/SilentSpeller/data/ark/*")