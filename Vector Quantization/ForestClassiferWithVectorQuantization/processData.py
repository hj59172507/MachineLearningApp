import os
import numpy as np

#use paths provided to find all file names for each class
def getDataFiles(paths):
	classCount = len(paths)
	dataFiles = []
	for i in range(classCount):
		dataFiles.append(set())

	for i in range(classCount):
		for root, dirs, files in os.walk(paths[i]):
			for file in files:
				if "txt" in file:
					dataFiles[i].add(root+'/'+file)
	return dataFiles

#specify a fix length and file path for all classes, open each file and cut it into piece of segment length
def getAllSegments(segmentLength, dataFiles):
	segments = []
	for i in range(len(dataFiles)):
		for file in dataFiles[i]:
			segments.extend(getSegment(file, segmentLength))
	return segments

#given a file name, return list of fixed length segment
def getSegment(file, segmentLength):
	segments, segment = [], []
	with open(file) as f:
		lines = f.readlines();
		size = segmentLength;
		for line in lines:
			if(size == 0):
				segments.append(segment)
				segment = []
				size = segmentLength
			segment.extend([int(x) for x in line.split()])
			size -= 1
	return segments

#if given source file exist, load and return it. Else compute from class path provided, save to disk and return it
def getSegmentsFromDisk(sourceFile, dataFiles, segmentLength):
	if os.path.isfile(sourceFile):
		return np.load(sourceFile)
	else:
		segmentsFromAllFiles = np.array(getAllSegments(segmentLength, dataFiles))
		np.save(sourceFile, segmentsFromAllFiles)
		return segmentsFromAllFiles

#given a paths to all classes, divide files in each class into n category
def splitDataFile(dataFiles, n):
	splitedDataFiles = [ [ [] for j in range(n) ] for i in range(len(dataFiles))]
	temp = 0
	for i in range(len(dataFiles)):
		for file in dataFiles[i]:
			if(temp == n):
				temp = 0
			splitedDataFiles[i][temp].append(file)
			temp += 1
	return splitedDataFiles

#given a splited data files, return a training set and testing set, where testing set is at index i as specified and training set is everything else
def getTrainAndTestSet(splitedDataFiles, testSetIndex):
	testSet, trainSet = [], []
	for i in range(len(splitedDataFiles)):
		testSet.append([])
		trainSet.append([])
		for j in range(len(splitedDataFiles[i])):
			if j == testSetIndex:
				testSet[i].extend(splitedDataFiles[i][j])
			else:
				trainSet[i].extend(splitedDataFiles[i][j])
	return testSet, trainSet




