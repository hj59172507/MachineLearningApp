import numpy as np

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
def splitData(pTest, test, train, dataSet):
	for d in dataSet:
		#random.random() return a random number between (0,1]
		randPTest = random.random()
		if(randPTest <= pTest):
			test.append(d)
		else:
			train.append(d)