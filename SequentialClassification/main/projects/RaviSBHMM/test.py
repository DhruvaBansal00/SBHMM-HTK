"""Defines method to train HMM and parser group to pass arguments to
train method. Can perform verification or recognition.

Methods
-------
test_cli
test
"""
import os
import glob
import shutil
from string import Template
from argparse import ArgumentParser, Namespace


def test_cli(parser: ArgumentParser) -> ArgumentParser:
    """Defines arguments to be passed to test() method.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to which to add the argument group for training.

    Returns
    -------
    return_val : ArgumentParser
        The parser with the testing argument group added.
    """

    group = parser.add_argument_group('test_args')
    group.add_argument('--start', type=int, default=-10)
    group.add_argument('--end', type=int, default=-1)
    group.add_argument('--method', default='recognition')

    return parser


def test(test_args) -> None:
    """Tests the HMM using HTK. Calls HVite and HResults. Can perform
    either recognition or verification. 

    Parameters
    ----------
    test_args : Namespace
        Argument group defined in test_cli() and split from main
        parser.
    """

    if os.path.exists('results'):
        shutil.rmtree('results')

    if os.path.exists('hresults'):
        shutil.rmtree('hresults')

    os.makedirs('results')
    os.makedirs('hresults')

    if test_args.end == -1:
        end = len(glob.glob('models/*hmm*'))

    if test_args.start < 0:
        start = end + test_args.start

    if test_args.method == 'recognition':

        # HVite_str = (f'HVite -A -H $macros -f -m -S lists/test.data -i $results '
        #              f'-p -10.0 -w wordNet.txt -s 25 dict wordList')
        HVite_str = (f'HVite -A -H $macros -m -S lists/test.data -i '
                     f'$results -w wordNet.txt -s 25 dict wordList')

        HVite_cmd = Template(HVite_str)

        HResults_str = (f'HResults -A -h -e \\?\\?\\? sil0 -e \\?\\?\\? '
                        f'sil1 -p -t -I all_labels.mlf wordList $results '
                        f'>> $hresults')
        HResults_cmd = Template(HResults_str)

    elif test_args.method == 'verification':

        HVite_str = (f'HVite -a -o N -T 1 -b sil0 sil1 -H $macros -S '
                     f'lists/test.data -i $results -m -y lab -t 250.0 -s 1.0 '
                     f'-p 0.0 -I all_labels.mlf -s 25 dict wordList')
        HVite_cmd = Template(HVite_str)
        HResults_cmd = Template('')

    elif test_args.method == 'alignment':

        HVite_str = (f'HVite -A -H $macros -m -f -S lists/test.data -i '
                     f'$results -w wordNet.txt -s 25 dict wordList')
        HVite_cmd = Template(HVite_str)
        HResults_cmd = Template('')

    for i in range(start, end):

        macros_filepath = f'models/hmm{i}/newMacros'
        results_filepath = f'results/res_hmm{i}.mlf'
        hresults_filepath = f'hresults/res_hmm{i}.txt'

        os.system(HVite_cmd.substitute(macros=macros_filepath,
                                       results=results_filepath))

        os.system(HResults_cmd.substitute(results=results_filepath,
                                          hresults=hresults_filepath))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = test_cli(parser)
    args = parser.parse_args()

    test(args)
