import os
import argparse
from get_confusion_matrix import get_confusion_matrix

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--confusion_matrix_filepath', type = str, default = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/hresults/')
    args = parser.parse_args()

    confusion_matrices = {}

    for i in range(7):
        confusion_matrix_filepath = os.path.join(args.confusion_matrix_filepath, str(i), 'res_hmm260.txt')
        confusion_matrix = get_confusion_matrix(confusion_matrix_filepath)
        confusion_matrices[confusion_matrix['user']] = confusion_matrix['matrix']
        # print(confusion_matrix)

    in_above_total = 0
    above_in_total = 0

    data = "user | above_in | in_above"
    for user in sorted(confusion_matrices.keys()):
        matrix = confusion_matrices[user]
        in_above = matrix['in']['above']
        above_in = matrix['abov']['in']
        in_above_total += in_above
        above_in_total += above_in
        data += f'\n{user} | {above_in} | {in_above}'
    data += f'\ntotal | {in_above_total} | {above_in_total} | {in_above_total + above_in_total}'
    print(data)

