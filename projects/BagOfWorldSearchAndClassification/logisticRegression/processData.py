import csv

def getDataWithoutStopingWord(file):
	with open(file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		line,documents, stars = 0, [], []
		for row in reader:
			if line == 0:
				line+=1
			else:
				documents.append(row[5].lower())
				stars.append(int(row[3]))
		return documents, stars

