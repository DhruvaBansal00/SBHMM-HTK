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

#print("Processing files in Extreme SSD")
#produceJSONS("/media/aslr/Extreme SSD/ProcessingPipeline/DATA/Videos/**/*.mkv", "/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Kinect_Data_July_2020")
print("Processing files")
produceJSONS("/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Videos/07-24-20_Matthew_4KDepth/*/*/*.mkv", "/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Kinect_Data_2020/07-24-20_Matthew_4KDepth")
