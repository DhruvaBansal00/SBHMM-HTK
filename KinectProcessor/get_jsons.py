import glob
import subprocess
import tqdm
import os

all_errors = "\nERRORS:\n"

def runOfflineProcessor(source: str, dest: str):
    
    global all_errors

    if " " in source:
        source = f"\"{source}\""
    if " " in dest:
        dest = f"\"{dest}\""
    command = f"./offline_processor_left {source} {dest}.json"
    try:
        output = subprocess.check_output(command, shell=True)
    except:
        print(f'{command} errored. please check {source}')
        all_errors += f'{command} errored. please check {source}\n'
    

def produceJSONS(source: str, destination: str):

    global all_errors

    video_list = glob.glob(source, recursive=True)

    for video in tqdm.tqdm(video_list):
        name = video.split("/")[-1].replace(".mkv", "")
        prefix = '/'.join(video.split("/")[-4:-1])
        destionationDir = os.path.join(destination, prefix)
        if not os.path.exists(destionationDir):
            os.makedirs(destionationDir)
            destinationFile = os.path.join(destination, prefix, name)
            # print(video, destinationFile, sep = ' | ')
            runOfflineProcessor(video, destinationFile)

    session = source.split('/')[-3]
    print(session)
    print(all_errors)
    f = open(f'Errors_{session}.txt', 'w')
    f.write(all_errors)
    f.close()

#
produceJSONS("/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Videos/10-02-20_ThadLeft_4KDepth/**/*.mkv", "/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Kinect_Data_2020/")


#print("Processing files in Extreme SSD")
#produceJSONS("/media/aslr/Extreme SSD/ProcessingPipeline/DATA/Videos/**/*.mkv", "/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Kinect_Data_July_2020")
# print("Processing files in U32 Shadow")
# produceJSONS("/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/Fingerspelling_Of_The_Dead/output_directory/Kinect/12-13-20_Matthew_Fingerspelling_Right/**/*.mkv", "/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/Fingerspelling_Of_The_Dead/output_directory/Kinect_json")
