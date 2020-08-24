import os, sys, glob

kinect_dir = '/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Mediapipe_Data_July_2020/08-15-20_Prerna_4K'
mediapipe_dir = '/media/aslr/U32 Shadow/ProcessingPipeline/DATA/Videos/08-15-20_Prerna_4K' #videos

kinect_filepaths = glob.glob(os.path.join(kinect_dir, '**'), recursive = True)
kinect_filepaths = list(filter(lambda filepath: os.path.isfile(filepath), kinect_filepaths))
kfps = []
for fp in kinect_filepaths:
    kfps.append('/'.join(fp.split('/')[-3:-1]))

mediapipe_filepaths = glob.glob(os.path.join(mediapipe_dir, '**'), recursive = True)
mediapipe_filepaths = list(filter(lambda filepath: os.path.isfile(filepath), mediapipe_filepaths))
mfps = []
for fp in mediapipe_filepaths:
    mfps.append('/'.join(fp.split('/')[-3:-1]))
    if '.mkv' not in fp: print("EXCEPTION: ", fp)


def Diff(list1, list2):
    return list(set(list(set(list1)-set(list2)) + list(set(list2)-set(list1))))

diffs = Diff(kfps, mfps)
phrases = {diff.split('/')[0] for diff in diffs}

print(len(diffs))
print(diffs[:5])
print(len(phrases))
print(sorted(phrases))