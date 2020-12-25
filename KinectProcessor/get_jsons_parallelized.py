import glob
import subprocess
import tqdm
import os

import multiprocessing
from ctypes import c_char_p

def runOfflineProcessorParallelized(source: str, dest: str, all_errors, lock):
    if " " in source:
        source = f"\"{source}\""
    if " " in dest:
        dest = f"\"{dest}\""
    command = f"./offline_processor {source} {dest}.json"
    try:
        output = subprocess.check_output(command, shell=True)
    except:
        lock.acquire()
        print(f'{command} errored. please check {source}')
        all_errors.value += f'{command} errored. please check {source}\n'   
        lock.release() 


def parallelizeHelper(video_list: list, destination: str, all_errors, lock):

    for video in tqdm.tqdm(video_list):
        name = video.split("/")[-1].replace(".mkv", "")
        prefix = '/'.join(video.split("/")[-4:-1])
        destionationDir = os.path.join(destination, prefix)
        if not os.path.exists(destionationDir):
            os.makedirs(destionationDir)
            destinationFile = os.path.join(destination, prefix, name)
            # print(video, destinationFile, sep = ' | ')
            runOfflineProcessorParallelized(video, destinationFile, all_errors, lock)
        

def parallelizeProduceJSONS(source: str, destination: str, num_processes = 3):
    processes = []
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    all_errors = manager.Value(c_char_p, "ERRORS:\n")

    video_list = glob.glob(source, recursive=True)
    start = end = 0
    
    for i in range(num_processes):
        end = start + (len(video_list) // num_processes)
        if i == num_processes - 1: end = len(video_list)

        video_split = video_list[start:end]
        start = end

        processes.append(multiprocessing.Process(target=parallelizeHelper, args=(video_split, destination, all_errors, lock)))
        
    for i in range(num_processes):
        processes[i].start()
    
    for i in range(num_processes):
        processes[i].join()

    # # all processes must be done by now, continue main process
    # session = source.split('/')[-3]
    # print(session)
    # print(all_errors.value)
    # f = open(f'Errors_{session}.txt', 'w')
    # f.write(all_errors.value)
    # f.close()
    


#
parallelizeProduceJSONS("/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Videos/08-12-20_Kanksha_4KDepth/**/*.mkv", "/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/CopyCatDatasetWIP/RawProcessingPipeline/DATA/Kinect_Data_2020/")


#print("Processing files in Extreme SSD")
#produceJSONS("/media/aslr/Extreme SSD/ProcessingPipeline/DATA/Videos/**/*.mkv", "/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Kinect_Data_July_2020")
# print("Processing files in U32 Shadow")
# produceJSONS("/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/Fingerspelling_Of_The_Dead/output_directory/Kinect/12-13-20_Matthew_Fingerspelling_Right/**/*.mkv", "/home/aslr/ccg-charizard/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/Fingerspelling_Of_The_Dead/output_directory/Kinect_json")
