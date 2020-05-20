from classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier
import sys
sys.path.insert(0, '../../')
from src.prepare_data.ark_reader import read_ark_files
import glob
import numpy as np


#structure -> word.name -> index -> state.name -> class
#supposed to create a map between word, index and the class number
def getClassTree(phrases):
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

def dataSetReader(classLabels, phrases, arkFileLoc):
    dataset = {}  ##Class to frames
    for phrase in phrases:
        currPhraseArk = arkFileLoc+phrase.name+".ark"
        content = read_ark_files(currPhraseArk)

        timeToFrame = content.shape[0]/phrase.end  ##aka frame rate

        for index, word in enumerate(phrase.words):
            for state in word.states:
                currClass = classLabels[word.name][index][state.name]
                
                # temp = content[int(state.start * timeToFrame) : int(state.end * timeToFrame)]
                # if temp.shape[0] == 1:
                #     print(temp.shape)
                if currClass in dataset:
                    dataset[currClass] = np.concatenate((dataset[currClass], content[int(state.start * timeToFrame) : int(state.end * timeToFrame)]))
                else:
                    dataset[currClass] = content[int(state.start * timeToFrame) : int(state.end * timeToFrame)]
        
    for dataClass in dataset:
        print(dataset[dataClass].shape)
        

def getTrainedClassifier(phrases):
    classLabels = getClassTree(phrases)
    dataSetReader(classLabels, phrases, "data/ark/")
    # print(classLabels)

