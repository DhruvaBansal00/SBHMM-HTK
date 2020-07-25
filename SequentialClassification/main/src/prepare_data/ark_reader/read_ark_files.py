import numpy as np

''' Returns data as numpy array '''
def read_ark_files(ark_file):
    with open(ark_file,'r') as ark_file:
        fileAsString = ark_file.read()
    contentString = fileAsString.split("[ ")[1].split("\n]")[0].split("\n")
    content = [[float(i) for i in frame.split()] for frame in contentString]
    return np.array(content)
