import csv

def getDataWithoutStopingWord(file):
	with open(file) as f:
		reader = csv.reader(f, delimiter=',')
		line,documents, stars = 0, [], []
		for row in reader:
			if line == 0:
				print(row)
				line+=1
			else:
				documents.append(row[5].lower())
				stars.append(int(row[3]))
		return documents, stars

