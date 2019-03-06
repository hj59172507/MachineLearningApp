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
segmentLength = 32
clusterCount = 480
allSegmentsFile, clusterModelLib = 'allSegments.npy', 'cluster.joblib'

dataFiles = pData.getDataFiles(paths)
aggCluster = getClusterModel(clusterModelLib, allSegmentsFile, dataFiles, segmentLength, clusterCount)
print(aggCluster.labels_)