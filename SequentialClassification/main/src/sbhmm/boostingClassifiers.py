from .classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.utils import shuffle
from joblib import Parallel, delayed

from src.prepare_data.ark_reader import read_ark_files
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

    
#structure -> word.name -> index -> state.name -> class
#supposed to create a map between word, index and the class number
def getClassTree(phrases: list, include_state: bool, include_index: bool) -> dict:
    wordToDict = {}
    currClass = 0
    for phrase in phrases:
        index = 0
        for word in phrase.words:
            for state in word.states:
                if word.name not in wordToDict:
                    wordToDict[word.name] = {}
                
                if include_state:

                    if include_index:
                
                        if index not in wordToDict[word.name]:
                            wordToDict[word.name][index] = {}
                        
                        if state.name not in wordToDict[word.name][index]:
                            wordToDict[word.name][index][state.name] = currClass
                            currClass += 1
                    
                    else:

                        if state.name not in wordToDict[word.name]:
                            wordToDict[word.name][state.name] = currClass
                            currClass += 1
                
                else:

                    if include_index:

                        if index not in wordToDict[word.name]:
                            wordToDict[word.name][index] = currClass
                            currClass += 1

                    else:

                        if type(wordToDict[word.name]) is dict:
                            wordToDict[word.name] = currClass
                            currClass += 1         

            index += 1
    
    print("Total classes = " + str(currClass))
    return wordToDict

def dataSetReader(classLabels: dict, phrases: list, arkFileLoc: str, include_state: bool, include_index: bool) -> dict:
    dataset = {}  ##Class to frames
    for phrase in phrases:
        currPhraseArk = arkFileLoc+phrase.name+".ark"
        content = read_ark_files(currPhraseArk)
        timeToFrame = content.shape[0]/phrase.end  ##aka frame rate

        for index, word in enumerate(phrase.words):
            for state in word.states:
                if include_index:
                    currClass = classLabels[word.name][index][state.name] if include_state else classLabels[word.name][index]

                else:
                    currClass = classLabels[word.name][state.name] if include_state else classLabels[word.name]

                if currClass in dataset:
                    dataset[currClass] = np.concatenate((dataset[currClass], content[int(state.start * timeToFrame) : int(state.end * timeToFrame)]))
                else:
                    dataset[currClass] = content[int(state.start * timeToFrame) : int(state.end * timeToFrame)]
        
    return dataset

def getDataSetForTrainingClass(dataset: dict, currClass: int) -> (list, list):
    features = []
    labels = []

    for classLabel in dataset:
        features.extend(dataset[classLabel])
        if classLabel == currClass:
            labels.extend([1 for i in range(dataset[classLabel].shape[0])])
        else:
            labels.extend([0 for i in range(dataset[classLabel].shape[0])])
    return np.array(features), np.array(labels)


def trainAdaboostClassifier(X, Y, seed):
    return AdaBoostClassifier(n_estimators=50, random_state=seed).fit(X, Y)

def calculateClassifierAcc(classifier: object, dataset: dict, trainMultipleClassifiers: bool):
    print("Calculating Classifier Accuracy")

    labels = [classLabel for classLabel in dataset]
    labels.sort()

    X = []
    Y = []
    for label in labels:
        for feature in dataset[label]:
            X.append(feature)
            Y.append(label)
    
    if trainMultipleClassifiers:
        transformation = np.zeros((len(X), len(classifier)))
        for i, unitClassifier in enumerate(classifier):
            transformation[:, i] = unitClassifier.predict_log_proba(X)[:, 1]
    
    else:
        transformation = classifier.predict_proba(X)
    
    predictions = np.argmax(transformation, axis=1)
    score = accuracy_score(Y, predictions)
    print("Classifier Accuracy is = " + str(score))
    return X, Y


def getTrainedClassifier(phrases: list, arkFileLoc: str, include_state: bool, include_index: bool, n_jobs: int, 
                        parallel: bool, knn_neighbors: int, classifierAlgo: str, trainMultipleClassifiers: bool = True, 
                        random_state: int = 42) -> object:
    classLabels = getClassTree(phrases, include_state, include_index)
    dataset = dataSetReader(classLabels, phrases, arkFileLoc, include_state, include_index)

    classifier = []

    if trainMultipleClassifiers:
        print("Training AdaBoosted Decision Tree Classifiers")
        if parallel:
            labels = [classLabel for classLabel in dataset]
            labels.sort()
            for iteration in tqdm(range(0, len(labels), n_jobs)):
                currLabels = [labels[i] for i in range(iteration, min(len(labels), iteration + n_jobs))]
                classifier += Parallel(n_jobs=len(currLabels))(delayed(trainAdaboostClassifier)(getDataSetForTrainingClass(dataset, currLabel)[0],
                            getDataSetForTrainingClass(dataset, currLabel)[1], random_state) for currLabel in currLabels)
        else:
            classifer = [AdaBoostClassifier(n_estimators=50, random_state=random_state) for classLabel in dataset]

            for classLabel in tqdm(dataset):
                X, Y = getDataSetForTrainingClass(dataset, classLabel)
                X, Y = shuffle(X, Y, random_state=random_state)
                classifer[classLabel].fit(X, Y)        
    else:
        features = []
        labels = []
        classes = [i for i in dataset]
        classes.sort()
        for classLabel in classes:
            features.extend(dataset[classLabel])
            labels.extend([classLabel for i in range(dataset[classLabel].shape[0])])
        features = np.array(features)
        labels = np.array(labels)
        if 'knn' in classifierAlgo:
            print("Training Master KNN Classifier")
            classifier = KNeighborsClassifier(n_neighbors=knn_neighbors)
        else:
            print("Training Master AdaBoost Classifier")
            classifier = AdaBoostClassifier(n_estimators=75, random_state=random_state)

        classifier.fit(features, labels)

    print("Classifier Training Completed")
    # X, Y = calculateClassifierAcc(classifier, dataset, trainMultipleClassifiers)
    # print("Train and Test Feature diff = " + str(features - X))
    # print("Train and Test Labels diff = " + str(labels - Y))
    return classifier
    

class ClassifierTransformer(object):
    
    def __init__(self, phrases, arkFileLoc, include_state, include_index, n_jobs, parallel, knn_neighbors, classifierAlgo, trainMultipleClassifiers=True, random_state=42):
        self.phrases = phrases
        self.trainMultipleClassifiers = trainMultipleClassifiers
        self.random_state = random_state

        self.classifier = getTrainedClassifier(self.phrases, arkFileLoc, include_state, include_index, n_jobs=n_jobs, parallel=parallel,
                        knn_neighbors=knn_neighbors, classifierAlgo=classifierAlgo, trainMultipleClassifiers=self.trainMultipleClassifiers,
                        random_state=self.random_state)
    
    def callDecisionFunction(self, index, features):
        return self.classifier[index].predict_log_proba(features)[:, 1]
    
    def getTransformedFeatures(self, features, parallel, n_jobs):

        if self.trainMultipleClassifiers:
            transformation = []
            transformation = np.zeros((features.shape[0], len(self.classifier)))
            for i in range(len(self.classifier)):
                transformation[:, i] = self.callDecisionFunction(i, features)                
            return np.array(transformation)
        else:
            return self.classifier.predict_proba(features)