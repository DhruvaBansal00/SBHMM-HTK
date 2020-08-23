import glob
import os
import tqdm


def runOfflineProcessor(source: str, dest: str):
    command = f"./offline_processor '{source}' '{dest}'.json"
    print(command)
    os.system(command)

def produceJSONS(source: str, destination: str):
    video_list = glob.glob(source, recursive=True)
    for video in tqdm.tqdm(video_list):
        name = ('/').join(video.split("/")[-3:]).replace(".mkv", "")
        destinationFile = os.path.join(destination, name)
        dirname = ('/').join(destinationFile.split('/')[:-1])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        runOfflineProcessor(video, destinationFile)

produceJSONS('/media/aslr/U32 Shadow/temp2/Kanksha/**/*.mkv', '/media/aslr/U32 Shadow/temp_jsons/Kanksha/')
