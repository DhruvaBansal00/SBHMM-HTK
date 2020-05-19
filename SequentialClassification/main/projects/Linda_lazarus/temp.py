
import os
import glob

htk_dir = os.path.join('data', 'htk')
if not os.path.exists(htk_dir):
    os.makedirs(htk_dir)

ark_dir = os.path.join('data', 'ark', '*.ark')
ark_files = glob.glob(ark_dir)

for ark_file in ark_files:

    print(ark_file)

    kaldi_command = (f'~/kaldi/src/featbin/copy-feats-to-htk '
                        f'--output-dir=data/htk '
                        f'--output-ext=htk '
                        f'--sample-period=40000 '
                        f'ark:{ark_file}')

    os.system(kaldi_command)