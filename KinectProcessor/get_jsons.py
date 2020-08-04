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
        name = video.split("/")[-1].strip(".mkv")
        destinationFile = os.path.join(destination, name)
        runOfflineProcessor(video, destinationFile)

produceJSONS('/home/dhruva/Desktop/CopyCat/**/*.mkv', '/home/dhruva/Desktop/CopyCat/Media/kinectJson')