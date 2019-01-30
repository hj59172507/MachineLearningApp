#thresholding matrix
thresholding(trainImgsOri, threshold, normalMin, normalMax)
thresholding(testImgsOri, threshold, normalMin, normalMax)

#shrink image and then stretch image
shrinkAndResize(trainImgsOri, trainImgsTouched, stretchDim)
shrinkAndResize(testImgsOri, testImgsTouched ,stretchDim)

#train for normal and original data
paras, classParas = [],[]
getParas(trainImgsOri, trainLabels, classCount, paras)
getClassParas(trainLabels, classCount, classParas)
correct = 0
for i in range(0,len(trainImgsOri)):
	if(predict(trainImgsOri[i],paras,classParas) == trainLabels[i]):
		correct += 1
print(f'Normal untouched using train: {correct / len(trainLabels)}')

correct = 0
for i in range(0,len(testImgsOri)):
	if(predict(testImgsOri[i],paras,classParas) == testLabels[i]):
		correct += 1
print(f'Normal untouched using test: {correct / len(testLabels)}')

#train for normal and streched data
paras, classParas = [],[]
getParas(trainImgsTouched, trainLabels, classCount, paras)
getClassParas(trainLabels, classCount, classParas)
correct = 0
for i in range(0,len(trainImgsTouched)):
	if(predict(trainImgsTouched[i],paras,classParas) == trainLabels[i]):
		correct += 1
print(f'Normal touched using test: {correct / len(trainLabels)}')

correct = 0
for i in range(0,len(testImgsTouched)):
	if(predict(testImgsTouched[i],paras,classParas) == testLabels[i]):
		correct += 1
print(f'Normal touched using test: {correct / len(testLabels)}')