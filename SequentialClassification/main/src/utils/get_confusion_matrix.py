import os

def get_confusion_matrix(results_file):
    results_dict = {'user': {}, 'general': {}, 'matrix': {}}
    
    labels = {}

    start_index = -1

    with open(results_file, 'r') as lf:
        start_file = 0
        for cnt, line in enumerate(lf):
            l = line.rstrip()

            if cnt == 1:
                labFile = l.split()[2]
                name = labFile.split('/')[3]
                session = name.split('.')[0]
                results_dict['user'] = session


            if start_file == 1:
                if "| Sum/Avg |" in l:
                    vals = l.split("|")[3].strip().split()
                    results_dict['general']['correct'] = float(vals[0])
                    results_dict['general']['sentence_error'] = float(vals[-1])

            if start_file == 2:
                letters = l.strip().split("   ")
                start_index = len(l) - len(l.lstrip())
                for i, letter in enumerate(letters):
                    labels[i] = letter

            if start_file == 2.5:
                for i, letter in enumerate(range(start_index, len(l), 4)):
                    if i in labels:
                        labels[i] += l[letter].strip()

            if start_file == 3:
                vals = l.split()
                x_label = vals[0]
                results_dict['matrix'][x_label] = {}

                vals = vals[1:len(labels)+1]

                for i, val in enumerate(vals):
                    results_dict['matrix'][x_label][labels[i]] = int(val)


            if "    ,-------------------------------------------------------------." in l:
                start_file = 1
            if start_file == 2:
                start_file = 2.5                
            if "------------------------ Confusion Matrix -------------------------" in l:
                start_file = 2
            if "[ %c / %e]" in l:
                labels[len(labels)] = 'del'
                start_file = 3

    return results_dict
