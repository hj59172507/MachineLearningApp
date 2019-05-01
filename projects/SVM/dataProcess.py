import numpy as np
import random

#take a txt file name, return only continuous variable as part of feature vector
def processData(fileName, featureVector, labels, continousFeatureIndexs):
	with open(fileName) as file:
		data = file.readlines()
		for line in data:
			features = line.split(',')
			featureVector.append([float(features[x]) for x in range(0,len(features)) if x in continousFeatureIndexs])
			label = features[len(features)-1].lstrip()
			if(label[0] == '<'):
				labels.append(-1)
			else:
				labels.append(1)

#return normalized data
def dataNormalization(featureVector):
	return (featureVector - np.mean(featureVector, axis = 0)) / np.std(featureVector, axis=0)

#split data into test and train with probability of pTest picking a datapoint for testset
def splitData(pTest, test, testLables, train, trainLables, dataSet, labels):
	for i in range(0, len(dataSet)):
		#random.random() return a random number between (0,1]
		randPTest = random.random()
		if(randPTest <= pTest):
			test.append(dataSet[i])
			testLables.append(labels[i])
		else:
			train.append(dataSet[i])
			trainLables.append(labels[i])

#randomly split data so that n data point will be in test, and the rest will be in train
def getRandomN(n, test, testLabels, train, trainLabels, dataSet, labels):
	pickForTest = random.sample(range(0,len(dataSet)),n)
	for i in range(0, len(dataSet)):
		if(i in pickForTest):
			test.append(dataSet[i])
			testLabels.append(labels[i])
		else:
			train.append(dataSet[i])
			trainLabels.append(labels[i])

#return a random sample and label
def getRandomSample(dataSet, labels):
	randomIndex = random.sample(range(0, len(dataSet)), 1)[0]
	return (dataSet[randomIndex], labels[randomIndex])

		