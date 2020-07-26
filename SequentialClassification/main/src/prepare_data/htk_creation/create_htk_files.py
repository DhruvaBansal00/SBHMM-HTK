"""Converts .ark files to .htk files for use by HTK.

Methods
-------
create_htk_files
"""
import os
import glob
import shutil
import tqdm



def create_htk_files(htk_dir: str = os.path.join('data', 'htk'), ark_dir: str = os.path.join('data', 'ark', '*.ark')) -> None:
    """Converts .ark files to .htk files for use by HTK.
    """
    if os.path.exists(htk_dir):
        shutil.rmtree(htk_dir)

    os.makedirs(htk_dir)

    ark_files = glob.glob(ark_dir)

    for ark_file in tqdm.tqdm(ark_files):
        
        kaldi_command = (f'~/kaldi/src/featbin/copy-feats-to-htk '
                         f'--output-dir={htk_dir} '
                         f'--output-ext=htk '
                         f'--sample-period=40000 '
                         f'ark:{ark_file}'
                         f'>/dev/null 2>&1')

        ##last line silences stdout and stderr

        os.system(kaldi_command)
