from .classes import State, Word, Phrase
from .adaEnsemble import AdaBoostedClassifierEnsemble

#get phrase assumes that the readline will return the path to the current phrase
def getPhrase(res_file: str) -> Phrase:
	phraseName = ".".join(res_file.readline().strip('\n').split("/")[-1].split(".")[:-1]) ##Extract name of the file
	words = []
	line = res_file.readline()
	currWord = None
	while line != ".\n":
		line_split = line.split()
		if len(line_split) > 4:
			#make new word
			if currWord != None:
				words.append(currWord)
			currWord = Word()
			currWord.start = float(line_split[0])
			currWord.name = line_split[4]

		currWord.states.append(State(float(line_split[0]), float(line_split[1]), line_split[2]))
		currWord.end = currWord.states[-1].end
		line = res_file.readline()
	
	if currWord != None:
		currWord.end = currWord.states[-1].end
		words.append(currWord)
	
	return Phrase(words[0].start, words[-1].end, phraseName, words)


def parse(res_file: str) -> list:
	pos = res_file.tell()
	nextline = res_file.readline()
	phrases = []
	while nextline != "":
		res_file.seek(pos)
		phrases.append(getPhrase(res_file))
		pos = res_file.tell()
		nextline = res_file.readline()
	return phrases
	

def getClassifierFromStateAlignment(resultFile: str, arkFolder: str = "data/ark/", include_state: bool = True, include_index: bool = True, n_jobs=4, parallel=False) -> object:

	curr_res_file = open(resultFile, "r")
	curr_res_file.readline()

	phrases = parse(curr_res_file)

	adaBoostedClassifier = AdaBoostedClassifierEnsemble(phrases, arkFolder, include_state, include_index, 
						trainMultipleClassifiers=True, random_state=42, n_jobs=n_jobs, parallel=parallel)
	return adaBoostedClassifier
