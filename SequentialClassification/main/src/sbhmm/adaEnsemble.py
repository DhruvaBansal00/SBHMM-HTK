from .classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier
import sys
from sklearn.utils import shuffle

from src.prepare_data.ark_reader import read_ark_files
import glob
import numpy as np
from tqdm import tqdm


    
#structure -> word.name -> index -> state.name -> class
#supposed to create a map between word, index and the class number
def getClassTree(phrases: list) -> dict:
    wordToDict = {}
    currClass = 0
    for phrase in phrases:
        index = 0
        for word in phrase.words:
            for state in word.states:
                if word.name not in wordToDict:
                    wordToDict[word.name] = {}
                
                if index not in wordToDict[word.name]:
                    wordToDict[word.name][index] = {}
                
                if state.name not in wordToDict[word.name][index]:
                    wordToDict[word.name][index][state.name] = currClass
                    currClass += 1
            index += 1

    print("Total classes = " + str(currClass))
    return wordToDict

def dataSetReader(classLabels: dict, phrases: list, arkFileLoc: str) -> dict:
    dataset = {}  ##Class to frames
    for phrase in phrases:
        currPhraseArk = arkFileLoc+phrase.name+".ark"
        content = read_ark_files(currPhraseArk)

        timeToFrame = content.shape[0]/phrase.end  ##aka frame rate

        for index, word in enumerate(phrase.words):
            for state in word.states:
                currClass = classLabels[word.name][index][state.name]

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
    
    return [np.array(features)[0]], [np.array(labels)[0]]



def getTrainedClassifier(phrases: list, arkFileLoc: str, trainMultipleClassifiers: bool = True, random_state: int = 42) -> object:
    classLabels = getClassTree(phrases)
    dataset = dataSetReader(classLabels, phrases, arkFileLoc)

    print("Starting classifier training")

    if trainMultipleClassifiers:
        classifer = [AdaBoostClassifier(n_estimators=100, random_state=random_state) for classLabel in dataset]

        for classLabel in tqdm(dataset):
            # print("Training binary classifier for class " + str(classLabel))
            X, Y = getDataSetForTrainingClass(dataset, classLabel)
            X, Y = shuffle(X, Y, random_state=random_state)
            classifer[classLabel].fit(X, Y)
            # print("Classifier " + str(classLabel) + " accepted training score = " + str(classifer[classLabel].score(dataset[classLabel], [1 for i in range(len(dataset[classLabel]))])))
            # print("Number accepted = "+str(len(dataset[classLabel])))
        
        print("Classifier Training Completed")
    else:
        features = []
        labels = []
        for classLabel in dataset:
            features.extend(dataset[classLabel])
            labels.extend([classLabel for i in range(dataset[classLabel].shape[0])])
        features = np.array(features)
        labels = np.array(labels)

        classifier = AdaBoostClassifier(n_estimators=100, random_state=random_state)
        classifier.fit(features, labels)
        
        for classLabel in dataset:
            labelDataset = [classLabel for i in range(len(dataset[classLabel]))]
            print(labelDataset)
            # print("Classifier " + str(classLabel) + " accepted training score = " + str(classifier.score(dataset[classLabel], [classLabel for i in range(len(dataset[classLabel]))])))
            # print("Number accepted = "+str(len(dataset[classLabel])))

    
    return classifer
    

class AdaBoostedClassifierEnsemble(object):
    
    def __init__(self, phrases, arkFileLoc, trainMultipleClassifiers=True, random_state=42):
        self.phrases = phrases
        self.trainMultipleClassifiers = trainMultipleClassifiers
        self.random_state = random_state

        self.classifier = getTrainedClassifier(self.phrases, arkFileLoc, trainMultipleClassifiers=self.trainMultipleClassifiers, random_state=self.random_state)
    
    def getTransformedFeatures(self, features):

        if self.trainMultipleClassifiers:
            transformation = np.zeros((features.shape[0], len(self.classifier)))
            for i in range(len(self.classifier)):
                transformation[:, i] = self.classifier[i].decision_function(features).flatten()
            return np.array(transformation)
        else:
            raise NotImplementedError("This feature hasn't been implemented since accuracies are really low")