import os

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
					dataFiles[i].add(file)
	return dataFiles
