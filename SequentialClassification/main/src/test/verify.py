"""Defines method to verify phases using HMM.

Methods
-------
verify
"""
import os
import glob
import shutil
from string import Template

def verification_cmd(model_iter: int, insertion_penalty: int, beam_threshold: int = 2000, fold: str = "", data_file: str):

    HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
                     f'lists/{fold}{data_file} -i $results -t {beam_threshold} '
                     f'-p {insertion_penalty} -I all_labels.mlf -s 25 dict wordList '
                     f'>/dev/null 2>&1')
    HVite_cmd = Template(HVite_str)
    macros_filepath = f'models/{fold}hmm{model_iter}/newMacros'
    results_filepath = f'results/{fold}res_hmm{model_iter}.mlf'

    os.system(HVite_cmd.substitute(macros=macros_filepath, results=results_filepath))


def verify(model_iter:int, insertion_penalty: int, beam_threshold: int = 2000, fold: str = "") -> None:
    
    