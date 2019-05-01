import processData as pData
import numpy as np
import os
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#relative path to each class folder
paths = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
#define fix size
segmentLength, clusterCount, splitCount, treeNum, maxDepths = 10, 36, 3, 30, 16
allSegmentsFile, clusterModelLib, rfModelLib = 'allSegments.npy', 'cluster.joblib', 'rfModel.joblib'
rfModelSaved = False

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

	dump(clf, rfModelLib)

	return clf.predict(testFV), testLabels

#given randomForest model and testing set, compute confusion table
def getConfusionMatrix(testSet):
	rfModel = load(rfModelLib)
	for i in range(len(testSet)):
		predictedLables = rfModel.predict(getFeatureVectors(kmeans, testSet[i]))
		predictedCount = [0 for j in range(len(testSet))]
		for lables in predictedLables:
			predictedCount[lables] += 1
		print(predictedCount)
		print(f'Error rate is: {1-predictedCount[i]/sum(predictedCount)}')
		print()

#given predicted and true labels, return accuracy
def getAccuracy(pLabels, tLables):
	correct = 0
	for i in range(len(pLabels)):
		if pLabels[i] == tLables[i]:
			correct += 1
	return correct / len(pLabels)

#plot a average histogram for all class
def plotHistForAllClasses(dataFiles, kmeans):
	x = np.arange(clusterCount)
	for i in range(len(dataFiles)):
		fvs = getFeatureVectors(kmeans, dataFiles[i]);
		meanFV = np.mean(np.array(fvs), 0)
		plt.bar(x, meanFV)
		plt.title(paths[i])
		plt.ylabel("count")
		plt.xlabel("k Clusters")
		plt.show()

#load cluster model
dataFiles = pData.getDataFiles(paths)
kmeans = getClusterModel(dataFiles)

#plot histogram for each classes
#plotHistForAllClasses(dataFiles, kmeans)

#train and test
splitedDataFiles = pData.splitDataFile(dataFiles, splitCount)
accuracies, avgAcc = [], 0
for i in range(splitCount):
	testSet, trainSet = pData.getTrainAndTestSet(splitedDataFiles, i)
	predictedLables, trueLabels = getLabelsUsingRandomForest(testSet, trainSet, kmeans)
	accuracies.append(getAccuracy(predictedLables, trueLabels))
avgAcc = sum(accuracies) / len(accuracies)

print(avgAcc)