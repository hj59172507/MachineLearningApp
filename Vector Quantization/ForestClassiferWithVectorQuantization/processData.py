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
def getSegmentsFromDisk(sourceFile, paths, segmentLength):
	if os.path.isfile(sourceFile):
		return np.load(sourceFile)
	else:
		dataFiles = pData.getDataFiles(paths)
		segmentsFromAllFiles = np.array(pData.getAllSegments(segmentLength, dataFiles))
		np.save(sourceFile, segmentsFromAllFiles)
		return segmentsFromAllFiles

