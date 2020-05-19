from classes import State, Word, Phrase
from sklearn.ensemble import AdaBoostClassifier


#structure -> word -> index -> state
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
                    

def getTrainedClassifier(phrases):
    classLabels = getClassTree(phrases)
    print(classLabels)
