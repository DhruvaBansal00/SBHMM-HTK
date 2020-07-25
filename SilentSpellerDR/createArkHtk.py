import os
import sys
import glob
import argparse
from tqdm import tqdm
import shutil

import numpy as np
import pandas as pd

def createArk(data: str, labels: str, saveDirectory: str) -> None:
    print("Creating ARK files")

    dataFiles = glob.glob(data)
    labelFiles = glob.glob(labels)

    if os.path.exists(saveDirectory):
        shutil.rmtree(saveDirectory)

    os.makedirs(saveDirectory)

    user = "Naoki"

    fileNames = {}

    for i in range(len(labelFiles)):
        with open(labelFiles[i], 'r') as currLabel:
            stamp = labelFiles[i].split("/")[-1].strip(".lab")
            letters = [i.strip("\n") for i in currLabel.readlines()][1:-1]
            fileNames[stamp] = saveDirectory+user+"."+"_".join(letters)+"."+stamp
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
    print("Creating HTK Files")

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

def createMLF(htk_dir: str, mlf_file: str) -> None:
    print("Create MLF File")

    filenames = glob.glob(htk_dir)

    with open(mlf_file, 'w') as f:
        
        f.write('#!MLF!#\n')

        for filename in tqdm(filenames):
            label = filename.split('/')[-1].replace('htk', 'lab')
            phrase = label.split('.')[1].split('_')
            f.write('"*/{}"\n'.format(label))
            f.write('sil0\n')

            for word in phrase:

                f.write('{}\n'.format(word.lower()))

            f.write('sil1\n')
            f.write('.\n')

if __name__ == "__main__":

    createArk('/home/dhruva/Desktop/CopyCat/SilentSpeller/gt2k_2328_16/data/*', '/home/dhruva/Desktop/CopyCat/SilentSpeller/gt2k_2328_16/label/*', "/home/dhruva/Desktop/CopyCat/SilentSpeller/data/ark/")
    createHTK('/home/dhruva/Desktop/CopyCat/SilentSpeller/data/ark/*', '/home/dhruva/Desktop/CopyCat/SilentSpeller/data/htk/')
    createMLF('/home/dhruva/Desktop/CopyCat/SilentSpeller/data/htk/*', '/home/dhruva/Desktop/CopyCat/SilentSpeller/all_labels.mlf')