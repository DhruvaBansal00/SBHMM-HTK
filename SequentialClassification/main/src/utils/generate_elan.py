import os
import csv
import shutil
from pympi.Elan import Eaf, to_eaf


def mlf_to_elan(mlf_filepath: str, video_dir: str, eaf_savedir: str) -> None:
    """Generates csv files from mlf file

    Parameters
    ----------
    eaf_filepath : str
        File path at which mlf file is located.

    video_dir : str
        Directory at which videos saved as mp4 are located.

    eaf_savedir : str
        Directory under which eaf files are saved.
    """

    '''
    eaf_file = Eaf("sample.eaf")
    eaf_file.add_linked_file("file:///Users/ishan/Documents/Research/annotation/alligator_above_blue_wagon.1.1596895580.mp4",
    relpath="./alligator_above_blue_wagon.1.1596895580.mp4",mimetype="video/mp4")
    '''

    # Iterate over lines of mlf file
    with open(mlf_filepath, "rb") as mlf:
        eaf_file = None
        out_path = None
        for line in mlf:
            line = str(line)

            # Move on to next eaf file if new file name is presented
            if not line[2].isdigit():

                # Save existing data to current eaf object
                if eaf_file:
                    to_eaf(out_path, eaf_file)

                # TODO: take eaf_savedir and append filename to create out_path
                out_path = "sample.eaf"

                # TODO: take video_dir and append filename to create video_filepath
                video_filepath = "/Users/ishan/Documents/Research/annotation/alligator_above_blue_wagon.1.1596895580.mp4"

                # Create base eaf file
                shutil.copy("elan.txt", out_path)

                # Create eaf object and link video
                eaf_file = Eaf(out_path)
                eaf_file.add_linked_file(video_dir, mimetype="video/mp4")

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