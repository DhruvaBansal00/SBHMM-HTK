class State:
	def __init__(self, start, end, name):
		self.start = start
		self.end = end
		self.name = name

class Word:
	def __init__(self):
		self.start = ""
		self.end = ""
		self.name = ""
		self.states = []

class Phrase:	
	def __init__(self, start, end, name, words):
		self.start = start
		self.end = end
		self.name = name
		self.words = words