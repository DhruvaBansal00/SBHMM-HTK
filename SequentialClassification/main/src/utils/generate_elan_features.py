import numpy as np
import os
import csv
import shutil
import glob
import re
from scipy.stats import norm 


def feature(x, components, feature_id):
    x = x[feature_id]
    max_ll = float('-inf')
    for mu, var in components:
        mu  = mu[feature_id]
        var = var[feature_id]
        ll  = norm.logpdf(x, mu, var)
        if ll > max_ll:
            max_ll = ll
    return max_ll


def model(model_path):
    models = {}
    current_label = None
    current_state = None
    current_vec   = None
    current_cmp   = []
    current_model = {}
    for line in open(model_path):
        if line.startswith('~h'):            
            if current_label is not None:
                models[current_label] = current_model
            current_model = {}
            current_label = line[4:-2]            
        elif line.startswith('<STATE>'):
            state_num = line.strip().split(' ')[-1]
            current_state = int(state_num)
            current_model[current_state] = []
        elif line.startswith('<MEAN>'):
            current_vec = 'mean'
        elif line.startswith('<VARIANCE>'):
            current_vec = 'variance'
        elif current_vec == 'mean':
            mean = np.array([float(x) for x in line.strip().split(' ')])
            current_cmp.append(mean)
            current_vec = None
        elif current_vec == 'variance':
            variances = np.array([float(x) for x in line.strip().split(' ')])
            current_cmp.append(variances)
            current_model[current_state].append(current_cmp)
            current_cmp = []
            current_vec = None
    if current_label is not None:
        models[current_label] = current_model
    return models        


def is_file_name(name: str) -> bool:
    return len(name)>0 and name.endswith("\"") and name[0]=="\""


def mlf_to_dict(mlf_filepath: str):
    '''Generates dictionary from mlf file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    Returns dictionary with this format:
    {filename : 
        {word : 
            [
                [state, start, end]
                ...
            ]
        }
        ...
    ...
    }
    '''
    out_dict = {}

    # Iterate over lines of mlf file
    with open(mlf_filepath, "rb") as mlf:
        out_path = None
        header = mlf.readline()
        lines = mlf.readlines()
        line_num = 0
        for line in lines:
            line = line.decode('utf-8').strip()

            # If line is file name, add new entry in dictionary
            if is_file_name(line):
                fname = '.'.join(line.split('/')[-1].split('.')[:-1])
                out_dict[fname] = {}

            # If line has state and boundary data
            elif line != '.':
                line_arr = line.split(" ")
                if len(line_arr) >= 5:
                    word = line_arr[4]
                    out_dict[fname][word] = []
                state = line_arr[2]
                start = int(line_arr[0])/1000
                end = int(line_arr[1])/1000
                out_dict[fname][word].append([state, start, end])                
        return out_dict

    
def features(path, hlist_path):
    os.system("chmod +x {}".format(path))
    process = os.popen("{} -i 100 {}".format(path, hlist_path))
    vectors = []
    for line in  process.read().split("\n")[1:-2]:    
        vector = [float(x) for x in re.split("\.[0-9]{3} *", line.split(":")[-1]) if x.strip() != ""]
        vectors.append(vector)
    return np.array(vectors)


def feature_importances(hmms, paths, inputs, hlist_path):
    annotations = []
    models  = model(hmms)
    results = paths
    for mlf_dir in os.listdir(results):
        results_dir = os.path.join(results, mlf_dir)
        if os.path.isdir(results_dir):
            for mlf in os.listdir(results_dir):
                data = mlf_to_dict(os.path.join(results, mlf_dir, mlf))
                for k, v in data.items():
                    path = "{}/{}.htk".format(inputs, k)            
                    vectors = features(path, hlist_path)
                    labels = []
                    for word, ranges in v.items():
                        for state, start, _ in ranges:
                            state = int(state[1:])
                            labels.append([word, state, int(start / 40)])
                    current_label = 0
                    for i, x in enumerate(vectors):
                        
                        if current_label + 1 < len(labels) and i >= labels[current_label + 1][-1]:
                            current_label += 1                    
                        w, s, _ = labels[current_label]
                        m = models[w][s]
                        # TODO dim 48 (HMM Macros) and features are 60 (HLIST)
                        for dim in range(0, int(min(len(x), len(m[0][0])))):
                            ll = feature(x, m, dim)
                            yield path, w, s, dim, ll
                        break
                    break


stream = feature_importances(    
    '../../projects/Kinect/models/9/hmm220/newMacros',
    '../../projects/Kinect/results/',
    '../../projects/Kinect/data/htk/',
    './HList'
)

with open('importance.csv', 'w') as fp:
    for p, w, s, dim, ll in stream:
        fp.write('{},{},{},{},{}\n'.format(p, w, s, dim, ll))
