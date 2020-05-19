from classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier
import sys
sys.path.insert(0, '../../')
from src.prepare_data.ark_reader import read_ark_files
import glob


#structure -> word -> index -> state
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

    # print("Total classes = " + str(currClass))
    return wordToDict

def dataSetReader(phrases, arkFileLoc):
    for phrase in phrases:
        currPhraseArk = arkFileLoc+phrase.name+".ark"
        content = read_ark_files(currPhraseArk)
        print(currPhraseArk)
        print(content.shape[0])
        print(phrase.end)
        print(content.shape[0]/phrase.end)
def getTrainedClassifier(phrases):
    classLabels = getClassTree(phrases)
    dataSetReader(phrases, "data/ark/")
    # print(classLabels)

