import glob
import subprocess
import tqdm
import os


def runOfflineProcessor(source: str, dest: str):
    if " " in source:
        source = f"\"{source}\""
    if " " in dest:
        dest = f"\"{dest}\""
    command = f"./offline_processor {source} {dest}.json"
    try:
        output = subprocess.check_output(command, shell=True)
    except:
        print(f'{command} errored. please check {source}')
    

def produceJSONS(source: str, destination: str):
    video_list = glob.glob(source, recursive=True)
    for video in tqdm.tqdm(video_list):
        name = video.split("/")[-1].replace(".mkv", "")
        destinationFile = os.path.join(destination, name)
        runOfflineProcessor(video, destinationFile)            

print("Processing files in Extreme SSD")
produceJSONS("/media/aslr/Extreme SSD/ProcessingPipeline/DATA/Videos/**/*.mkv", "/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Kinect_Data_July_2020")
print("Processing files in U32 Shadow")
produceJSONS("/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Videos/**/*.mkv", "/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Kinect_Data_July_2020")
