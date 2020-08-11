import glob
import os
import tqdm


def runOfflineProcessor(source: str, dest: str):
    command = f"./offline_processor {source} {dest}.json"
    print(command)
    os.system(command)

def produceJSONS(source: str, destination: str):
    video_list = glob.glob(source, recursive=True)
    for video in tqdm.tqdm(video_list):
        name = video.split("/")[-1].replace(".mkv", "")
        destinationFile = os.path.join(destination, name)
        runOfflineProcessor(video, destinationFile)

produceJSONS('/mnt/ExtremeSSD/4K-kinect-recordings/kinect/full4-kinect/*.mkv', '/mnt/ExtremeSSD/ProcessingPipeline/DATA/Kinect_Data_July_2020')
