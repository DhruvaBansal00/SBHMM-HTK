import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import argparse

def make_features_table(data_dir, users, save_dir, mode):

    # data_dir: the mediapipe data set directory

    # save_dir: the location where the tables are saved (actual directory will be <save_dir>/tables/<mode>)

    # mode options: [trials, phrases, words, all].
    #  - trials: creates a table for each unique trial
    #  - phrases: creates a table for each unique phrase
    #  - words: creates a table for each unique word. Note that this option in reality combines the data from each trial that contains "word"
    #  - all: creates a table for the entire dataset

    # Return images of tables in the save_dir location  

    # make output path

    if not (mode == 'trials' or mode == 'phrases' or mode == 'words' or mode == 'all'):
        return

    tables_dir = os.path.join(save_dir, 'visualization', 'tables', mode)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    os.chdir(tables_dir)

    data_files_list = []
    for user in users:
        data_files_list += glob.glob(os.path.join(data_dir, '*/*/*{}*.data'.format(user)))
    if not users:
        data_files_list = glob.glob(os.path.join(data_dir, '*/*/*.data'))

    # to hold the score
    boxes_score_dict = {'boxes': {}, 'landmarks': {}, 'faces': {}}

    if mode == 'trials':

        for data_file in data_files_list:
            
            print("Creating table for {}".format(data_file))

            # get frequnecy of data for current data file 
            result_table = make_table(data_file)

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            table_file_name = data_file.split('/')[-1].replace('.data', '')
            table_dir = os.path.join(data_file.split('/')[-3], data_file.split('/')[-2])
            #print(table_file_name, table_dir)

            make_plot(result_table, table_file_name, table_dir = table_dir)

            # score among trials between [0, 1]
            boxes_score_dict['boxes'][table_file_name] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][table_file_name] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][table_file_name] = round(result_table[1][2], 4) # 1_face            
    
    elif mode == 'phrases':

        phrase_dict = {}
        for data_file in data_files_list:
            phrase = data_file.split('/')[-1].split('.')[1]
            if phrase not in phrase_dict:
                phrase_dict[phrase] = []
            phrase_dict[phrase].append(data_file)

        for phrase in phrase_dict:
            result_table = np.array([[0 for i in range(3)] for j in range(4)])
            for data_file in phrase_dict[phrase]:
                # get frequnecy of data for current data file 
                current_table = make_table(data_file)
                result_table += current_table 

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            make_plot(result_table, phrase)

            # score among phrases between [0, 1]
            boxes_score_dict['boxes'][phrase] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][phrase] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][phrase] = round(result_table[1][2], 4) # 1_face
            
    elif mode == 'words':
        word_dict = {}
        for data_file in data_files_list:
            phrase = data_file.split('/')[-1].split('.')[1]
            words = phrase.split("_")
            for word in words:
                if word not in word_dict:
                    word_dict[word] = []
                word_dict[word].append(data_file)

        for word in word_dict:
            result_table = np.array([[0 for i in range(3)] for j in range(4)])
            for data_file in word_dict[word]:

                # get frequnecy of data for current data file 
                current_table = make_table(data_file)
                result_table += current_table 

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            make_plot(result_table, word)

            # score among words between [0, 1]
            boxes_score_dict['boxes'][word] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][word] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][word] = round(result_table[1][2], 4) # 1_face

    elif mode == 'all':
        result_table = np.array([[0 for i in range(3)] for j in range(4)])
        for data_file in data_files_list:

            # get frequnecy of data for current data file
            current_table = make_table(data_file)
            result_table += current_table 

        # find percent of data detected and round to 4 decimal places
        data_length = np.sum(result_table, axis=0)[0]
        result_table = np.divide(result_table, float(data_length))
        result_table = np.round(result_table, 4)

        # make string table
        make_plot(result_table, "full_summary")

        # score among dataset between [0, 1]
        boxes_score_dict['boxes'][mode] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
        boxes_score_dict['landmarks'][mode] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
        boxes_score_dict['faces'][mode] = round(result_table[1][2], 4) # 1_face
        print(result_table)

    boxes_score_dict['boxes'] = sorted(boxes_score_dict['boxes'].items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    boxes_score_dict['landmarks'] = sorted(boxes_score_dict['landmarks'].items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    boxes_score_dict['faces'] = sorted(boxes_score_dict['faces'].items(), key = lambda kv:(kv[1], kv[0]), reverse = True)


    # print(boxes_score_dict)

    # write the boxes score to a file in the respective tables directory
    with open("{}_score.json".format(mode), "w") as file:
        file.write(json.dumps(boxes_score_dict, indent=4)) 


def _load_json(json_file):
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
    return data

def make_table(data_file):
    curr_data = _load_json(data_file)
    if not curr_data:
        return None
    curr_data = {int(key): value for key, value in curr_data.items()}

    table = np.array([[0 for i in range(3)] for j in range(4)])

    # populate table by frequency of occurence
    for i in range(len(curr_data)):
        if not curr_data[i]["boxes"] or sum(curr_data[i]["boxes"]["0"][0:4]) == 0:
            table[0][0] += 1
        elif len(curr_data[i]["boxes"]) > 2:
            table[3][0] += 1
        else:
            table[len(curr_data[i]["boxes"])][0] += 1

        if not curr_data[i]["landmarks"]:
            table[0][1] += 1
        elif len(curr_data[i]["landmarks"]) > 2:
            table[3][1] += 1
        else:
            table[len(curr_data[i]["landmarks"])][1] += 1
        
        if not curr_data[i]["faces"]:
            table[0][2] += 1
        elif len(curr_data[i]["faces"]) > 2:
            table[3][2] += 1
        else:
            table[len(curr_data[i]["faces"])][2] += 1
    return table

def make_plot(table, table_title, table_dir = ""):

    # make string table
    strtable = [['' for i in range(3)] for j in range(4)]
    for i in range(4):
        for j in range(3):
            strtable[i][j] = str(table[i][j])

    # make plot
    columns = ['Hand Boxes', 'Hand Landmarks', 'Face']
    bar_width = 0.4
    y_offset = np.zeros(len(columns))

    p1 = plt.bar(columns, table[0], bar_width, bottom=y_offset, color='#CC0000')
    y_offset = y_offset + table[0]
    p2 = plt.bar(columns, table[1], bar_width, bottom=y_offset, color='#FFCC00')
    y_offset = y_offset + table[1]
    p3 = plt.bar(columns, table[2], bar_width, bottom=y_offset, color='#009900')
    y_offset = y_offset + table[2]
    p4 = plt.bar(columns, table[3], bar_width, bottom=y_offset, color='#006699')

    plt.legend((p4[0], p3[0], p2[0], p1[0]), (">2", "2", "1", "0"), loc='center right', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    plt.table(cellText=strtable, rowLabels=["0", "1", "2", ">2"], colLabels=columns, loc='bottom', bbox=[0, -0.4, 1, 0.25], cellLoc='center')

    plt.title(label=table_title)

    if len(table_dir) and not os.path.exists(table_dir):
        os.makedirs(table_dir)
        
    save_dir = os.path.join(table_dir, table_title)

    plt.savefig('{}.png'.format(save_dir)) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_features_dir', default = '/media/thad/Seagate Backup Plus Drive/July_Mediapipe_Features')
    parser.add_argument('--users', default=['Prerna'])
    parser.add_argument('--base_project_dir', default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/Prerna_Interpolation_HMMs')
    parser.add_argument('--mode', default = 'all')
    args = parser.parse_args()

    make_features_table(args.base_features_dir, args.users, args.base_project_dir, args.mode)