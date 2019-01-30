import csv
import numpy as np
import random
from scipy.stats import norm

#define some constant
diabetePositive = 1
diabeteNegative = 0
omitCols = [2,3,5,7]

#read and store data
def readCSV(file, dataSet):
	with open(file) as csv_file:
		csv_reader = csv.reader(csv_file)
		for line in csv_reader:
			dataSet.append([float(attr) for attr in line])


#split data into test and train with probability of pTest picking a datapoint for testset
def splitData(pTest, test, train, dataSet):
	for d in dataSet:
		#random.random() return a random number between (0,1]
		randPTest = random.random()
		if(randPTest <= pTest):
			test.append(d)
		else:
			train.append(d)

#find normal distribution parameter for given data
def getNormPara(dataSet, diaParas, nonDiaParas, featureCount, omit = False):
	diabeteSet, nonDiabeteSet = [],[]
	for i in range(0, featureCount):
		diabeteSet.append([])
		nonDiabeteSet.append([])
	#group same features into label group
	for dataPoint in dataSet:
		for i in range(0,featureCount):
			if( not (omit and (i in omitCols) and dataPoint[i]==0) ):
				if(dataPoint[featureCount-1] == diabetePositive):
					diabeteSet[i].append(dataPoint[i])
				else:
					nonDiabeteSet[i].append(dataPoint[i])
	for i in range(0,featureCount-1):
		diaParas[i] = [np.mean(diabeteSet[i]), np.std(diabeteSet[i])]
		nonDiaParas[i] = [np.mean(nonDiabeteSet[i]), np.std(nonDiabeteSet[i])]
	pDiabetes = len(diabeteSet[0])/len(dataSet)
	diaParas[featureCount-1].append(pDiabetes)
	nonDiaParas[featureCount-1].append(1-pDiabetes)
	
#predict datapoint using given normal distribution parameters
def predict(dataPoint, diaParas, nonDiaParas, omit = False):
	pDia,pNonDia = 0,0
	featureCount = len(dataPoint)-1
	for i in range(0, featureCount):
		if( not (omit and (i in omitCols) and dataPoint[i]==0) ):
			pDia += np.log(norm.pdf(dataPoint[i], diaParas[i][0], diaParas[i][1]))
			pNonDia += np.log(norm.pdf(dataPoint[i], nonDiaParas[i][0], nonDiaParas[i][1]))
	if( (pDia + np.log(diaParas[featureCount])) > (pNonDia + np.log(nonDiaParas[featureCount])) ):
		return diabetePositive
	else:
		return diabeteNegative

#set up data
dataSet = []
file = 'pima-indians-diabetes.csv'
readCSV(file, dataSet)
featureCount = len(dataSet[0])

#perform train and test
numSplit = 10
pTest = 0.2
accuracy = []
for i in range(0, numSplit):
	test, train, diaParas, nonDiaParas = [],[],[],[]
	for i in range(0, featureCount):
		diaParas.append([])
		nonDiaParas.append([])
	#split data
	splitData(pTest, test, train, dataSet)
	#get distribution parameter on train set
	getNormPara(train, diaParas, nonDiaParas, featureCount)
	#make prediction of test set 
	correctCount = 0
	for t in test:
		result = predict(t, diaParas, nonDiaParas)
		if(result == t[len(t)-1]):
			correctCount += 1
	accuracy.append(correctCount/len(test))

for a in accuracy:
	print(a)
print(f'Average accuracy is : {np.mean(accuracy)}')

#perform train and test while omitting zero value for 3rd, 4th, 6th and 8th column
accuracy = []
for i in range(0, numSplit):
	test, train, diaParas, nonDiaParas = [],[],[],[]
	for i in range(0, featureCount):
		diaParas.append([])
		nonDiaParas.append([])
	#split data
	splitData(pTest, test, train, dataSet)
	#get distribution parameter on train set
	getNormPara(train, diaParas, nonDiaParas, featureCount, True)
	#make prediction of test set 
	correctCount = 0
	for t in test:
		result = predict(t, diaParas, nonDiaParas, True)
		if(result == t[len(t)-1]):
			correctCount += 1
	accuracy.append(correctCount/len(test))

for a in accuracy:
	print(a)
print(f'Average accuracy after omitting 0 value for column 3,4,6,8 is : {np.mean(accuracy)}')