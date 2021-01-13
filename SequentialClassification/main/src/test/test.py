"""Defines method to test HMM. Can perform recognition.

Methods
-------
test
"""
import os
import glob
import shutil
from string import Template

def test(start: int, end: int, method: str, insertion_penalty: int, beam_threshold: int = 2000, fold: str = "") -> None:
    """Tests the HMM using HTK. Calls HVite and HResults. Can perform
    either recognition or verification. 

    Parameters
    ----------
    test_args : Namespace
        Argument group defined in test_cli() and split from main
        parser.
    """

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if method != 'alignment':

        if os.path.exists(f'hresults/{fold}'):
            shutil.rmtree(f'hresults/{fold}')
        os.makedirs(f'hresults/{fold}')
    
    if end == -1:
        end = len(glob.glob(f'models/{fold}*hmm*'))

    if start < 0:
        start = end + start

    if method == 'recognition':

        # HVite_str = (f'HVite -A -H $macros -f -m -S lists/test.data -i $results '
        #              f'-p -10.0 -w wordNet.txt -s 25 dict wordList')
        HVite_str = (f'HVite -A -H $macros -m -S lists/{fold}test.data -i '
                     f'$results -p {insertion_penalty} -w wordNet.txt -s 25 dict wordList')

        HVite_cmd = Template(HVite_str)

        HResults_str = (f'HResults -A -h -e \\?\\?\\? sil0 -e \\?\\?\\? '
                        f'sil1 -p -t -I all_labels.mlf wordList $results '
                        f'>> $hresults')
        HResults_cmd = Template(HResults_str)

    elif method == 'alignment':

        HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
                     f'lists/{fold}train.data -i $results -t {beam_threshold} '
                     f'-p {insertion_penalty} -I all_labels.mlf -s 25 dict wordList '
                     f'>/dev/null 2>&1')
        HVite_cmd = Template(HVite_str)
        HResults_cmd = Template('')

    for i in range(start, end):

        macros_filepath = f'models/{fold}hmm{i}/newMacros'
        results_filepath = f'results/{fold}res_hmm{i}.mlf'
        hresults_filepath = f'hresults/{fold}res_hmm{i}.txt'

        os.system(HVite_cmd.substitute(macros=macros_filepath,
                                       results=results_filepath))

        os.system(HResults_cmd.substitute(results=results_filepath,
                                          hresults=hresults_filepath))
