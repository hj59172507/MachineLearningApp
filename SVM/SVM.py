import numpy as np
import dataProcess
import random

#calculate the cost 
def getCost(x, y, a, b, lam):
	a = np.array(a)
	cost = y*(np.dot(a,x)+b)+(lam*np.dot(a,a.T))/2
	return max(0, 1-cost)

#calculate accuracy for given dataset
def getTempAcc(a, b, testSet, testLabels):
	correct = 0
	for i in range(0, len(testSet)):
		ytemp = np.dot(a,testSet[i]) + b
		if(ytemp*testLabels[i]>=0):
			correct += 1
	return correct / len(testSet)

#calcualte gradiant and update
def gradiantDecent(dataSet, labels, a, b, lambdas, learnRate, lam):
	cost = getCost(x, y, a[lam], b[lam], lambdas[lam])
	if(cost != 0):
		a[lam] -= np.dot(learnRate,(np.dot(a[lam],lambdas[lam])-np.dot(x,y)))
		b[lam] += learnRate*y
	else:
		a[lam] -= np.dot(learnRate,np.dot(a[lam],lambdas[lam]))

#defien variables for configuration and storage
continousFeatureIndexs = [0,2,4,10,11,12]
trainFileName, testFileName = 'train.txt', 'test.txt'
trainFV, testFV, trainLabels, testLabels = [], [], [], []

dataProcess.processData(trainFileName, trainFV, trainLabels, continousFeatureIndexs)
dataProcess.processData(testFileName, testFV, testLabels, continousFeatureIndexs)

trainFV = dataProcess.dataNormalization(trainFV)
testFV = dataProcess.dataNormalization(testFV)

#define constants for training SVM, this SVM use sign(ax + b) to make prediction
lambdas = [0.0001, 0.001, 0.01, 0.1, 1]
aVectors = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
bVector = [1,1,1,1,1]
batchSize, seasons, stepsPerSeason, accTestNumber, confirmAccSteps = 1, 50, 300, 50, 30
tempAccs = [[],[],[],[],[]]

#split the train data to 90-10 for training and validating
trainModelFV, testModelFV, trainModelLabels, testModelLabels = [], [], [], []
ptest = 0.1
dataProcess.splitData(ptest, testModelFV, testModelLabels, trainModelFV, trainModelLabels, trainFV, trainLabels)

#loop to train models build with different lambdas
for season in range(1, seasons+1):
	learnRate = 100 / (season+10000)
	trainFVDiv, testFV50, trainLabsDiv, testLabs50 = [],[], [], []
	dataProcess.getRandomN(accTestNumber, testFV50, testLabs50, trainFVDiv, trainLabsDiv, trainModelFV, trainModelLabels)
	for step in range(1, stepsPerSeason+1):
		#update each model use
		for lam in range(0, len(lambdas)):
			x,y = dataProcess.getRandomSample(trainFVDiv, trainLabsDiv)
			gradiantDecent(x, y, aVectors, bVector, lambdas, learnRate, lam)
			if(step % 30 == 0):
				tempAccs[lam].append(getTempAcc(aVectors[lam], bVector[lam], testFV50, testLabs50))

for i in range(0, len(tempAccs)):
	size = len(tempAccs[i])
	print(f'last ten acc: {tempAccs[i][size-10:size-1]} for lambda {lambdas[i]}')

