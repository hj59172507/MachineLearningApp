import os

#relative path to each class folder
paths = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']

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
def getAllSegment(segmentLength, dataFiles):
	segments = []
	for i in range(len(dataFiles)):
		for file in dataFiles[i]:
			with open(file) as f:
				lines = f.readlines();
				size = segmentLength;
				segment = []
				for line in lines:
					if(size == 0):
						segments.append(segment)
						segment = []
						size = segmentLength
					segment.extend([int(x) for x in line.split()])
					size -= 1
	return segments

#dummy varible to check result
dataFiles = getDataFiles(paths)
segments = getAllSegment(32,dataFiles)