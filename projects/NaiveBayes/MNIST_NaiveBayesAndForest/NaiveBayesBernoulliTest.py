#thresholding matrix
thresholding(trainImgsOri, threshold, beruMin, beruMax)
thresholding(testImgsOri, threshold, beruMin, beruMax)

#shrink image and then stretch image
shrinkAndResize(trainImgsOri, trainImgsTouched, stretchDim)
shrinkAndResize(testImgsOri, testImgsTouched ,stretchDim)

#train for bernoulli and original data
paras, classParas = [],[]
getParas(trainImgsOri, trainLabels, classCount, paras, bern = True)
getClassParas(trainLabels, classCount, classParas)
correct = 0
for i in range(0,len(trainImgsOri)):
	if(predict(trainImgsOri[i],paras,classParas, bern = True)== trainLabels[i]):
		correct += 1
print(f'bernoulli untouched using train: {correct / len(trainLabels)}')

correct = 0
for i in range(0,len(testImgsOri)):
	if(predict(testImgsOri[i],paras,classParas, bern = True) == testLabels[i]):
		correct += 1
print(f'bernoulli untouched using test: {correct / len(testLabels)}')

#train for bernoulli and streched data
paras, classParas = [],[]
getParas(trainImgsTouched, trainLabels, classCount, paras, bern = True)
getClassParas(trainLabels, classCount, classParas)
correct = 0
for i in range(0,len(trainImgsTouched)):
	if(predict(trainImgsTouched[i],paras,classParas, bern = True) == trainLabels[i]):
		correct += 1
print(f'bernoulli touched using test: {correct / len(trainLabels)}')

correct = 0
for i in range(0,len(testImgsTouched)):
	if(predict(testImgsTouched[i],paras,classParas, bern = True) == testLabels[i]):
		correct += 1
print(f'bernoulli touched using test: {correct / len(testLabels)}')