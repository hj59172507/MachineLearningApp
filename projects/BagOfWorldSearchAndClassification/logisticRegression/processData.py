import csv
import random

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

#split data into test and train with probability of testProb picking a datapoint for testset
def splitData(dataSet, testProb, labels):
	test, testLabels, train, trainLabels = [],[],[],[]
	for i in range(len(dataSet)):
		randPTest = random.random()
		if(randPTest <= testProb):
			testLabels.append(labels[i])
			test.append(dataSet[i])
		else:
			trainLabels.append(labels[i])
			train.append(dataSet[i])
	return test, testLabels, train, trainLabels
