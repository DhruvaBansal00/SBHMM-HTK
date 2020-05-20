from classes import State, Word, Phrase
from adaEnsemble import AdaBoostedClassifierEnsemble

#get phrase assumes that the readline will return the path to the current phrase
def getPhrase(res_file: str) -> Phrase:
	phraseName = res_file.readline().strip('\n').split("/")[-1].split(".")[0].split("_") ##Extract the file of the ark file with underscores instead of dots
	phraseName = phraseName[0]+"."+"_".join(phraseName[1:-1])+"."+phraseName[-1] ##Add dots to match ark file format
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
	

def getClassifierFromStateAlignment(resultFile: str, arkFolder: str = "data/ark/") -> object:

	curr_res_file = open(resultFile, "r")
	curr_res_file.readline()

	phrases = parse(curr_res_file)

	# for phrase in phrases:
	# 	print(phrase.name)
	# 	for word in phrase.words:
	# 		print(word.name)
	# 		for state in word.states:
	# 			print(state.name + " " + str(state.start) + " " + str(state.end))	

	adaBoostedClassifier = AdaBoostedClassifierEnsemble(phrases, arkFolder, trainMultipleClassifiers=True, random_state=42)
	return adaBoostedClassifier
