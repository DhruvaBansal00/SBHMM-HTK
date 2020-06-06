import os
import sys
import glob
import argparse
from tqdm import tqdm
import shutil

import numpy as np
import pandas as pd

def createArk(data: str, labels: str) -> None:
    dataFiles = glob.glob(data)
    labelFiles = glob.glob(labels)
    saveDirectory = "/home/dhruva/Desktop/CopyCat/SilentSpeller/ARK/16/"

    if os.path.exists(saveDirectory):
        shutil.rmtree(saveDirectory)

    os.makedirs(saveDirectory)

    user = "Naoki"

    fileNames = {}

    for i in range(len(labelFiles)):
        with open(labelFiles[i], 'r') as currLabel:
            stamp = labelFiles[i].split("/")[-1].strip(".lab")
            letters = [i.strip("\n") for i in currLabel.readlines()][1:-1]
            fileNames[stamp] = saveDirectory+user+"."+"sil0_"+"_".join(letters)+"_sil1."+stamp
            currLabel.close()

    for i in tqdm(range(len(dataFiles))):
        with open(dataFiles[i], 'r') as currData:
            newFileName = fileNames[dataFiles[i].split("/")[-1]]
            arkTitle = newFileName.split("/")[-1]
            newFileName += ".ark"
            newContent = [arkTitle, " ", "[", " "]
            newContent.extend([i.strip(" \n")+"\n" for i in currData.readlines()])
            newContent.extend("]")
            currData.close()

            with open(newFileName, 'w') as newArkFile:
                newArkFile.writelines(newContent)
                newArkFile.close()


def createHTK(ark_dir: str, htk_dir: str) -> None:

    if os.path.exists(htk_dir):
        shutil.rmtree(htk_dir)

    os.makedirs(htk_dir)

    ark_files = glob.glob(ark_dir)

    for ark_file in tqdm(ark_files):
        
        kaldi_command = (f'~/kaldi/src/featbin/copy-feats-to-htk '
                         f'--output-dir={htk_dir} '
                         f'--output-ext=htk '
                         f'--sample-period=40000 '
                         f'ark:{ark_file}'
                         f'>/dev/null 2>&1')

        ##last line silences stdout and stderr

        os.system(kaldi_command)

def createMLF(htk_dir: str) -> None:
    


createArk('/home/dhruva/Desktop/CopyCat/SilentSpeller/PCA/naoki_pca/16/*', '/home/dhruva/Desktop/CopyCat/SilentSpeller/PCA/naoki_pca/label/*')
createHTK('/home/dhruva/Desktop/CopyCat/SilentSpeller/ARK/16/*', '/home/dhruva/Desktop/CopyCat/SilentSpeller/HTK/16/')