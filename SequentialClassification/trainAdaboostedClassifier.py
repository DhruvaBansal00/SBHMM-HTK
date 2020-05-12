from classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier


def getClassTree(phrases):
    wordToDict = {}
    currClass = 0
    for phrase in phrases:
        for word in phrase.words:
            for state in word.states:
                if word.name not in wordToDict:
                    wordToDict[word.name] = {}
                
                if state.name not in wordToDict[word.name]:
                    wordToDict[word.name][state.name] = currClass
                    currClass += 1

    print("Total classes = " + str(currClass))
    return wordToDict
                    

def getTrainedClassifier(phrases):
    classLabels = getClassTree(phrases)
    print(classLabels)
