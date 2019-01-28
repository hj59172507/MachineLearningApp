from sklearn.ensemble import RandomForestClassifier

#train using forest
treeNumbers, maxDepths = [10, 30], [4, 16]
trainImgsOriOneD, testImgsOriOneD, trainImgsTouOneD, testImgsTouOneD = [],[],[],[]
oneDDimL, oneDDimS = (1, 28*28), (1, 20*20)
arrayToMatrix(trainImgsOri, trainImgsOriOneD, oneDDimL, reduceDim = True)
arrayToMatrix(testImgsOri, testImgsOriOneD, oneDDimL, reduceDim = True)
arrayToMatrix(trainImgsTouched, trainImgsTouOneD, oneDDimS, reduceDim = True)
arrayToMatrix(testImgsTouched, testImgsTouOneD, oneDDimS, reduceDim = True)


for treeNumber in treeNumbers:
	for maxDepth in maxDepths:
		for data in dataToUse:			

			clf = RandomForestClassifier(n_estimators = treeNumber, max_depth = maxDepth)
			clfTrainlabels, clfTestlabels = [],[]
			trainMessage, testMessage = '',''
			trainCorrect, testCorrect = 0, 0
			if(data == original):
				clf.fit(trainImgsOriOneD, trainLabels)			
				clfTrainlabels = clf.predict(trainImgsOriOneD)
				clfTestlabels = clf.predict(testImgsOriOneD)
				trainMessage = f'Accuracy for Forest classifier with {treeNumber} trees, {maxDepth} maximum Depth, untouched, train data: '
				testMessage = f'Accuracy for Forest classifier with {treeNumber} trees, {maxDepth} maximum Depth, untouched, test data: '
			else:
				clf.fit(trainImgsTouOneD, trainLabels)
				clfTrainlabels = clf.predict(trainImgsTouOneD)
				clfTestlabels = clf.predict(testImgsTouOneD)
				trainMessage = f'Accuracy for Forest classifier with {treeNumber} trees, {maxDepth} maximum Depth, touched, train data: '
				testMessage = f'Accuracy for Forest classifier with {treeNumber} trees, {maxDepth} maximum Depth, touched, test data: '

			for i in range(0,len(clfTrainlabels)):
				if(clfTrainlabels[i] == trainLabels[i]):
					trainCorrect += 1
			print(trainMessage + f'{trainCorrect / len(trainLabels)}')
			
			for i in range(0,len(clfTestlabels)):
				if(clfTestlabels[i] == testLabels[i]):
					testCorrect += 1
			print(testMessage + f'{testCorrect / len(testLabels)}')
	