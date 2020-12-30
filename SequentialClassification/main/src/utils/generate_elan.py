import os
import csv
import shutil
import glob
from pympi.Elan import Eaf, to_eaf


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
        
def make_elan(data: dict, has_states: bool, video_dirs: list, eaf_savedir: str) -> None:
    """Generates eaf files from data dict

    Parameters
    ----------
    data : dict
        Segmentation data extracted from mlf files.

    has_states : bool
        Whether or not to write individual states to eaf.

    video_dirs : list[str]
        List of videos to create eaf objects with.

    eaf_savedir : str
        Directory under which eaf files are saved.
    """
    video_names = [ '.'.join(vname.split('/')[-1].split('.')[:-1]) for vname in video_dirs ]
    for fname in data:
        if fname in video_names:
            video_fp = video_dirs[video_names.index(fname)]

            out_path = os.path.join(eaf_savedir, fname + '.eaf')

            # Create base eaf file
            shutil.copy("elan.txt", out_path)

            # Create eaf object and link video
            eaf_file = Eaf(out_path)
            eaf_file.add_linked_file(video_fp.replace(' ', '\ '), mimetype="video/mp4")

            # Iterate over segmentation data
            if not has_states:
                eaf_file.add_tier(fname)
                for word in data[fname]:
                    start = int(data[fname][word][0][1])
                    end = int(data[fname][word][-1][2])
                    eaf_file.add_annotation(fname, start, end, word)
            else:
                for word in data[fname]:
                    eaf_file.add_tier(word)
                    for state in data[fname][word]:
                        state_num = state[0]
                        start = state[1]
                        end = state[2]
                        eaf_file.add_annotation(word, int(start), int(end), state_num)
            
            # Create eaf out of data
            to_eaf(out_path, eaf_file)

def mlf_to_elan(mlf_filepath: str, video_dirs: list, eaf_savedir: str) -> None:
    """Generates eaf files from mlf file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    video_dirs : list[str]
        List of videos to create eaf objects with.

    eaf_savedir : str
        Directory under which eaf files are saved.
    """

    # Iterate over lines of mlf file
    with open(mlf_filepath, "rb") as mlf:
        eaf_file = None
        out_path = None
        video_fp = None

        lines = mlf.readlines()
        line_num = 1
        while line_num < len(lines):
            line = str(lines[line_num])
            updated = False

            if len(line) < 10:
                line_num += 1
                continue

            # Move on to next eaf file if new file name is presented
            elif not line[2].isdigit():

                # Save existing data to current eaf object
                if eaf_file:
                    to_eaf(out_path, eaf_file)

                # create filename out of header info
                fname = line.split('/')[-1][:-8]

                # take eaf_savedir and append filename to create out_path
                out_path = os.path.join(eaf_savedir, fname + '.eaf')

                # check if mlf has corresponding video
                for name in video_dirs:
                    if fname == name.split('/')[-1][:-4]:
                        video_fp = name
                        updated = True
                        break
                if not updated:
                    line_num += 1
                    while line_num < len(lines) and len(str(lines[line_num])) > 10:
                        line_num += 1
                    updated = True
                    continue

                # Create base eaf file
                shutil.copy("elan.txt", out_path)

                # Create eaf object and link video
                eaf_file = Eaf(out_path)
                eaf_file.add_linked_file(video_fp.replace(' ', '\ '), mimetype="video/mp4")

            # Gather data from mlf and add tiers, annotations, and start/end times
            else:
                line_arr = line[2:-3].split(" ")
                if len(line_arr) >= 5:
                    word = line_arr[4]
                    eaf_file.add_tier(word)
                state = line_arr[2]
                start = line_arr[0]
                end = line_arr[1]
                eaf_file.add_annotation(word, int(int(start)/1000), int(int(end)/1000), state)
            
            line_num+=1
        
        # Save existing data to current eaf object
        if eaf_file:
            to_eaf(out_path, eaf_file)


if __name__=='__main__':

    # Find where videos are located on desktop
    video_dirs = glob.glob('/media/aslr/disk1/Video_Backup_MP4/**/*.mp4', recursive=True)

    # Save annotated videos on desktop
    save_dir = '/media/aslr/disk1/video_annotation'
    '''
    # Iterate over MLF files
    results = '../../projects/Kinect/results/'
    mlf_dirs = [ os.path.join(results, mlf) for mlf in os.listdir(results) if os.path.isdir(os.path.join(results, mlf)) ]
    for idx, mlf_dir in enumerate(mlf_dirs):
        print('Progress: ' + str(idx / len(mlf_dirs)))
        for mlf in os.listdir(mlf_dir):
            mlf_to_elan(os.path.join(mlf_dir, mlf), video_dirs, save_dir)
    '''
    results = '../../projects/Kinect/results/'
    for mlf_dir in os.listdir(results):
        path = os.path.join(os.getcwd(), results, mlf_dir)
        if os.path.isdir(path):
            for mlf in os.listdir(path):
                data = mlf_to_dict(os.path.join(path, mlf))
                make_elan(data, True, video_dirs, save_dir)
