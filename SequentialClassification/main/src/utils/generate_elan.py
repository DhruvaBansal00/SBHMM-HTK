import os
import csv
import shutil
import glob
from pympi.Elan import Eaf, to_eaf


def mlf_to_elan(mlf_filepath: str, video_dirs: list, eaf_savedir: str) -> None:
    """Generates csv files from mlf file

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
    video_dirs = glob.glob('/media/thad/DataBackup/Video_Backup_MP4/**/*.mp4', recursive=True)

    # Save annotated videos on desktop
    save_dir = '/media/thad/DataBackup/video_annotation'

    # Iterate over MLF files
    results = '../../projects/Kinect/results/'
    mlf_dirs = [ os.path.join(results, mlf) for mlf in os.listdir(results) if os.path.isdir(os.path.join(results, mlf)) ]
    for idx, mlf_dir in enumerate(mlf_dirs):
        print('Progress: ' + str(idx / len(mlf_dirs)))
        for mlf in os.listdir(mlf_dir):
            mlf_to_elan(os.path.join(mlf_dir, mlf), video_dirs, save_dir)
