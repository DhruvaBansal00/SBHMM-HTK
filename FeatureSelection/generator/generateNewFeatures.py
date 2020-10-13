from classifierFromState import getClassifierFromStateAlignment
import glob
import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

def read_ark_files(ark_file):
    with open(ark_file,'r') as ark_file:
        fileAsString = ark_file.read()
    contentString = fileAsString.split("[ ")[1].split("\n]")[0].split("\n")
    content = [[float(i) for i in frame.split()] for frame in contentString]
    return np.array(content)

def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    if not os.path.exists(os.path.dirname(ark_filepath)):
        os.makedirs(os.path.dirname(ark_filepath))

    with open(ark_filepath, 'w+') as out:
        out.write('{} [ '.format(title))
        
    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')

    with open(ark_filepath, 'a') as out:
        out.write(']')


def createNewArkFiles(reduced_all_data_frames: list, arkFileData: dict, arkFileSave: str):

    for arkFile in tqdm.tqdm(arkFileData):

        arkFileName = arkFile.split("/")[-1]
        arkFileSavePath = arkFileSave + arkFileName
        newContent = reduced_all_data_frames[arkFileData[arkFile][0]:arkFileData[arkFile][1]]
        _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))

def generateFeatures(resultFile: str, arkFolder: str, classifier: str, include_state: bool, include_index: bool,
								 n_jobs: int, parallel: bool, trainMultipleClassifiers: bool, knn_neighbors: float,
                                 generated_features_folder: str, pca_components: int, no_pca: bool) -> object:
    
    trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFolder, classifier=classifier, include_state=include_state, 
                        include_index=include_index, n_jobs=n_jobs, parallel=parallel, trainMultipleClassifiers=trainMultipleClassifiers,
                        knn_neighbors=int(knn_neighbors))
    
    arkFiles = glob.glob(arkFolder+"/*")


    arkFileData = {}
    all_data_frames = []
    curr_index = 0

    print(f'Transforming .ark files')
    
    for arkFile in tqdm.tqdm(arkFiles):
        curr_content = read_ark_files(arkFile)
        arkFileData[arkFile] = [curr_index, curr_index + len(curr_content)]
        all_data_frames.extend(trainedClassifier.getTransformedFeatures(curr_content, parallel, n_jobs))
        curr_index += len(curr_content)
    
    if not no_pca:
        print(f'PCAing .ark files')
        pca = PCA(n_components=pca_components)
        all_data_frames = pca.fit_transform(all_data_frames)

    print(f'Writing .ark files to {generated_features_folder}')
    createNewArkFiles(all_data_frames, arkFileData, generated_features_folder)

def getFeatureImportance(resultFile: str, arkFolder: str, classifier: str, include_state: bool, include_index: bool,
								 n_jobs: int, parallel: bool, trainMultipleClassifiers: bool, knn_neighbors: float) -> None:
    
    trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFolder, classifier=classifier, include_state=include_state, 
                        include_index=include_index, n_jobs=n_jobs, parallel=parallel, trainMultipleClassifiers=trainMultipleClassifiers,
                        knn_neighbors=int(knn_neighbors))
    
    feature_importance = [[i, score] for i,score in enumerate(trainedClassifier.classifier.feature_importances_)]
    feature_importance.sort(key=lambda x: x[1])
    
    print(feature_importance)


getFeatureImportance("../alignment/0/res_hmm240.mlf", "../data/ark/", "adaboost", True, False, 1, False, False, 0)