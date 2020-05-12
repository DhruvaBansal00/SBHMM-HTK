class state:
	def __init__(self, start, end, name):
		self.start = start
		self.end = end
		self.name = name

class word:
	def __init__(self, start, end, name, states):
		self.start = start
		self.end = end
		self.name = name
		self.states = states

class phrase:	
	def __init__(self, start, end, name, words):
		self.start = start
		self.end = end
		self.name = name
		self.words = words

def getPhrase(res_file):
	return 0

def parse(res_file):
	for i in res_file:
		print(i)

curr_res_file = open("sample_res_hmm.mlf", "r")
curr_res_file.readline()
parse(curr_res_file)