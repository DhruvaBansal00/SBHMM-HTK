import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import argparse
import shutil

def make_features_table(input_mediapipe_directory: str, users: list, output_project_directory: str, mode: str):

    # input_mediapipe_directory: the mediapipe data set directory

    # output_project_directory: the location where the tables are saved (actual directory will be <save_dir>/tables/<mode>)

    # mode options: [trials, phrases, words, all].
    #  - trials: creates a table for each unique trial
    #  - phrases: creates a table for each unique phrase
    #  - words: creates a table for each unique word. Note that this option in reality combines the data from each trial that contains "word"
    #  - all: creates a table for the entire dataset

    # Return images of tables in the save_dir location  

    # make output path

    if not (mode == 'trials' or mode == 'phrases' or mode == 'words' or mode == 'all'):
        return

    base_tables_directory = os.path.join(output_project_directory, 'visualization/tables', mode)

    if not os.path.exists(base_tables_directory):
        os.makedirs(base_tables_directory)

    if not users:
        features_filepaths = glob.glob(os.path.join(input_mediapipe_directory, '**/*.data'), recursive = True)
    else:
        features_filepaths = []
        for user in users:
            features_filepaths.extend(glob.glob(os.path.join(input_mediapipe_directory, '*{}*'.format(user), '**/*.data'), recursive = True))

    # to hold the score
    boxes_score_dict = {'boxes': {}, 'landmarks': {}, 'faces': {}}

    if mode == 'trials':

        for features_filepath in features_filepaths:
            
            print("Creating table for {}".format(features_filepath))

            # get frequnecy of data for current data file 
            result_table = make_table(features_filepath)

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            feature_filename = features_filepath.split('/')[-1]
            session, phrase, trial = feature_filename.split('.')[0:3]
            table_filename = '{}.{}.{}.png'.format(session, phrase, trial)
            table_directory = os.path.join(base_tables_directory, session, phrase, trial)
            make_plot(result_table, table_filename = table_filename, table_directory = table_directory)

            # score among trials between [0, 1]
            boxes_score_dict['boxes'][table_filename] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][table_filename] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][table_filename] = round(result_table[1][2], 4) # 1_face
    
    elif mode == 'phrases':

        phrase_dict = {}
        for features_filepath in features_filepaths:
            
            phrase = features_filepath.split('/')[-1].split('.')[1]

            try: phrase_dict[phrase].append(features_filepath)
            except KeyError: phrase_dict[phrase] = [features_filepath]

        for phrase in phrase_dict:
            result_table = np.array([[0 for i in range(3)] for j in range(4)])
            for features_filepath in phrase_dict[phrase]:
                # get frequnecy of data for current data file 
                current_table = make_table(features_filepath)
                result_table += current_table
                feature_filename = features_filepath.split('/')[-1]
                session, phrase, trial = feature_filename.split('.')[0:3]

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            table_filename = '{}.{}.png'.format(session, phrase)
            table_directory = os.path.join(base_tables_directory, session, phrase)
            make_plot(result_table, table_filename = table_filename, table_directory = table_directory)

            # score among phrases between [0, 1]
            boxes_score_dict['boxes'][phrase] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][phrase] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][phrase] = round(result_table[1][2], 4) # 1_face
            
    elif mode == 'words':

        word_dict = {}
        for features_filepath in features_filepaths:

            phrase = features_filepath.split('/')[-1].split('.')[1]
            words = phrase.split("_")
            for word in words:

                try: word_dict[word].append(features_filepath)
                except KeyError: word_dict[word] = [features_filepath]

        for word in word_dict:
            result_table = np.array([[0 for i in range(3)] for j in range(4)])
            for features_filepath in word_dict[word]:

                # get frequnecy of data for current data file 
                current_table = make_table(features_filepath)
                result_table += current_table 

            # find percent of data detected and round to 4 decimal places
            data_length = np.sum(result_table, axis=0)[0]
            result_table = np.divide(result_table, float(data_length))
            result_table = np.round(result_table, 4)

            # make plot for table
            table_filename = '{}.{}.png'.format(session, word)
            table_directory = os.path.join(base_tables_directory, session, word)
            make_plot(result_table, table_filename = table_filename, table_directory = table_directory)

            # score among words between [0, 1]
            boxes_score_dict['boxes'][word] = round(result_table[1][0] * 0.5 + result_table[2][0], 4) # 1_hand * 0.5 + 2_hand
            boxes_score_dict['landmarks'][word] = round(result_table[1][1] * 0.5 + result_table[2][1], 4) # 1_landmark * 0.5 + 2_landmark
            boxes_score_dict['faces'][word] = round(result_table[1][2], 4) # 1_face

    elif mode == 'all':
        result_table = np.array([[0 for i in range(3)] for j in range(4)])
        for features_filepath in features_filepaths:

            # get frequnecy of data for current data file
            current_table = make_table(features_filepath)
            result_table += current_table 

        # find percent of data detected and round to 4 decimal places
        data_length = np.sum(result_table, axis=0)[0]
        result_table = np.divide(result_table, float(data_length))
        result_table = np.round(result_table, 4)
        feature_filename = features_filepaths[0].split('/')[-1]
        session, phrase, trial = feature_filename.split('.')[0:3]

        # make string table
        table_filename = '{}.png'.format(session)
        table_directory = os.path.join(base_tables_directory, session)
        make_plot(result_table, table_filename = table_filename, table_directory = table_directory)

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
    score_filepath = os.path.join(base_tables_directory, session, '{}_score.json'.format(mode))
    with open(score_filepath, "w") as file:
        file.write(json.dumps(boxes_score_dict, indent=4)) 


def _load_json(json_file):
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
    return data

def make_table(features_filepath):
    curr_data = _load_json(features_filepath)
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

def make_plot(table, table_filename, table_directory):

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

    plt.title(label=table_filename)

    if len(table_directory) and not os.path.exists(table_directory):
        os.makedirs(table_directory)
        
    save_dir = os.path.join(table_directory, table_filename)

    plt.savefig('{}'.format(save_dir)) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mediapipe_directory', type = str, default = '/mnt/ExtremeSSD/ProcessingPipeline/DATA/Mediapipe_Data_July_2020')
    parser.add_argument('--users', nargs='*', default = None)
    parser.add_argument('--output_project_directory', type = str, default = '/home/thad/copycat/SBHMM-HTK/SequentialClassification/main/projects/July2020Mediapipe')
    parser.add_argument('--mode', type = str, default = 'all')
    args = parser.parse_args()

    make_features_table(args.input_mediapipe_directory, args.users, args.output_project_directory, args.mode)