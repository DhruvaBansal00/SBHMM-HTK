import glob

def runOfflineProcessor(source: str, dest: str):
    

def produceJSONS(source: str, destination: str):
    video_list = glob.glob(source, recursive=True)
    print(video_list)

produceJSONS('/home/dhruva/Desktop/CopyCat/**/*.mkv', '/home/dhruva/Desktop/CopyCat/Media/kinectJson')