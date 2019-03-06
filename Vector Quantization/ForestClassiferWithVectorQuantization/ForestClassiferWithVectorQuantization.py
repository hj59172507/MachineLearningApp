import processData as pData
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from joblib import dump, load

#load and return cluster model if already computed before, else compute, save and return the model
def getClusterModel(clusterModelLib, allSegmentsFile, dataFiles, segmentLength, clusterCount):
	if os.path.isfile(clusterModelLib):
		return load(clusterModelLib)
	else:
		segmentsFromAllFiles = pData.getSegmentsFromDisk(allSegmentsFile, dataFiles, segmentLength)
		aggCluster = AgglomerativeClustering(n_clusters = clusterCount).fit(segmentsFromAllFiles)
		dump(aggCluster, clusterModelLib)
		return aggCluster

#relative path to each class folder
paths = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
#define fix size
segmentLength, clusterCount, splitCount = 32, 480, 3
allSegmentsFile, clusterModelLib = 'allSegments.npy', 'cluster.joblib'

#load cluster model
dataFiles = pData.getDataFiles(paths)
aggCluster = getClusterModel(clusterModelLib, allSegmentsFile, dataFiles, segmentLength, clusterCount)

#load train and test set
splitedDataFiles = pData.splitDataFile(dataFiles, splitCount)
testSet1, trainSet1 = pData.getTrainAndTestSet(splitedDataFiles, 0)
testSet2, trainSet2 = pData.getTrainAndTestSet(splitedDataFiles, 1)
testSet3, trainSet3 = pData.getTrainAndTestSet(splitedDataFiles, 2)

print(aggCluster.labels_)