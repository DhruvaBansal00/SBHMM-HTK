import os
import csv
import shutil
from pympi.Elan import Eaf, to_eaf

data_dict = {}
#eaf2 = Eaf("research1.eaf")

shutil.copy("elan.txt", "sample.eaf")

eaf_file = Eaf("sample.eaf")
eaf_file.add_linked_file("file:///Users/ishan/Documents/Research/annotation/alligator_above_blue_wagon.1.1596895580.mp4",
    relpath="./alligator_above_blue_wagon.1.1596895580.mp4",mimetype="video/mp4")


with open("hmm160_sample.mlf", "rb") as mlf:
    word = ""
    state = ""
    for line in mlf:
        line = str(line)[2:-3] if len(str(line))>4 else str(line)
        if line[0].isdigit():
            line_arr = line.split(" ")
            if len(line_arr) >= 5:
                word = line_arr[4]
                data_dict[word] = {}
                eaf_file.add_tier(word)
                print(word)
            state = line_arr[2]
            start = line_arr[0]
            end = line_arr[1]
            data_dict[word][state] = [start, end]
            eaf_file.add_annotation(word, int(int(start)/1000), int(int(end)/1000), state)

with open("hmm160_sample.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    for key1, value1 in data_dict.items():
        for key2, value2 in value1.items():
            value2 = [ int(val) / 1000 for val in value2 ]
            writer.writerow([key1, key2, value2[0], value2[1]])

to_eaf("/Users/ishan/Documents/Research/annotation/sample.eaf", eaf_file)