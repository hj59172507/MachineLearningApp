import processData as pData
import numpy as np
import os
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

#relative path to each class folder
paths = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
#define fix size
segmentLength, clusterCount, splitCount, treeNum, maxDepths = 32, 48, 3, 30, 16
allSegmentsFile, clusterModelLib = 'allSegments.npy', 'cluster.joblib'

#load and return cluster model if already computed before, else compute, save and return the model
def getClusterModel(dataFiles):
	if os.path.isfile(clusterModelLib):
		return load(clusterModelLib)
	else:
		segmentsFromAllFiles = pData.getSegmentsFromDisk(allSegmentsFile, dataFiles, segmentLength)
		kmeans = KMeans(n_clusters = clusterCount).fit(segmentsFromAllFiles)
		dump(kmeans, clusterModelLib)
		return kmeans

#given a sample files, return histogram vectors from cluster model
def getFeatureVectors(kmeans, files):
	fVectors = []
	for file in files:
		fVector = [0 for i in range(kmeans.n_clusters)]
		segments = pData.getSegment(file, segmentLength)
		for label in kmeans.predict(segments):
			fVector[label] += 1
		fVectors.append(fVector)
	return fVectors

#given training and testing set, return predicted and true labels
def getLabelsUsingRandomForest(testSet, trainSet, kmeans):
	testFV, trainFV, testLabels, trainLabels, = [],[],[],[]

	for i in range(len(testSet)):
		testFV.extend(getFeatureVectors(kmeans, testSet[i]))
		testLabels.extend([i for j in range(len(testSet[i]))])
		trainFV.extend(getFeatureVectors(kmeans, trainSet[i]))
		trainLabels.extend([i for j in range(len(trainSet[i]))])

	clf = RandomForestClassifier(n_estimators = treeNum, max_depth = maxDepths)
	clf.fit(trainFV, trainLabels)
	return clf.predict(testFV), testLabels

#given predicted and true labels, return accuracy
def getAccuracy(pLabels, tLables):
	correct = 0
	for i in range(len(pLabels)):
		if pLabels[i] == tLables[i]:
			correct += 1
	return correct / len(pLabels)

#load cluster model
dataFiles = pData.getDataFiles(paths)
kmeans = getClusterModel(dataFiles)

#train and test
splitedDataFiles = pData.splitDataFile(dataFiles, splitCount)
accuracies, avgAcc = [], 0
for i in range(splitCount):
	testSet, trainSet = pData.getTrainAndTestSet(splitedDataFiles, i)
	predictedLables, trueLabels = getLabelsUsingRandomForest(testSet, trainSet, kmeans)
	accuracies.append(getAccuracy(predictedLables, trueLabels))
avgAcc = sum(accuracies) / len(accuracies)

print(avgAcc)