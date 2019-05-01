import cv2
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from scipy.stats import norm
from scipy.stats import bernoulli
from sklearn.ensemble import RandomForestClassifier

#thresholding image, for any value less than threshold, set to min, else set to max
def thresholding(imgs, threshold, min, max):
	for img in imgs:
		for row in range(0, len(img)):
			low = img[row] < threshold
			high = img[row] >= threshold
			img[row][low] = min
			img[row][high] = max

#convert all arrays to matrixs with given dimension
def arrayToMatrix(src, tar, dim, reduceDim = False):
	for arr in src:
		if(reduceDim):
			tar.append(np.reshape(arr, dim)[0])
		else:
			tar.append(np.reshape(arr, dim))

#get min,max value of non-zero row index
def getRowMinMax(img):
	min, max = len(img)//2, len(img)//2
	for row in range(0,len(img)):
		if(np.max(img[row]) != 0):
			if(row < min):
				min = row
			if(row > max):
				max = row
	return min,max

#first shrink image to get rid of all zero row and column, then resized back to resizedDim
def shrinkAndResize(imgs, touchedImgs, resizeDim):
	for img in imgs:
		yMin, yMax = getRowMinMax(img)
		xMin, xMax = getRowMinMax(img.T)
		shrinkImg = img[yMin:yMax+1].T[xMin:xMax+1].T
		resizeImg = cv2.resize(shrinkImg, resizeDim, interpolation=cv2.INTER_NEAREST)
		touchedImgs.append(resizeImg)

#find normal or bernoulli distribution parameter 
def getParas(imgs, labels, classCount, paras, bern = False):
	classMat = []
	for i in range(0, classCount):
		paras.append([])
		classMat.append([])
	for i in range(0, len(imgs)):
		classMat[labels[i]].append(imgs[i])
	for i in range(0,classCount):
		if(bern):
			paras[i] = np.mean(classMat[i], axis = 0)
		else:
			paras[i] = (np.mean(classMat[i], axis = 0), np.std(classMat[i], axis = 0))

#get class probability
def getClassParas(labels, classCount, classParas):
	for i in range(0, classCount):
		classParas.append(0)
	for label in labels:
		classParas[label] += 1
	for i in range(0, classCount):
		classParas[i] /= len(labels)

#predict base on bayes parameters and return class label
def predict(img, paras, classParas, bern = False):
	resultSet = []
	for i in range(0, len(classParas)):
		resultSet.append(0)
	for classIndex in range(0, len(classParas)):
		if(bern):
			resultSet[classIndex] = np.nansum(np.log(bernoulli.pmf(img, paras[classIndex])))
		else:
			resultSet[classIndex] = np.nansum(np.log(norm.pdf(img, paras[classIndex][0], paras[classIndex][1])))
	for i in range(0, len(classParas)):
		resultSet[i] += np.log(classParas[i])
	return np.argmax(resultSet)


#load data
mndata = MNIST()
mndata.gz = True
trainImgs, trainLabels = mndata.load_training()
testImgs, testLabels = mndata.load_testing()

#set up constant
original, stretched = 0, 1
dataToUse = [original, stretched]
threshold = 100
oriDim = (28,28)
stretchDim = (20,20)
classCount = 10
normalMin = 0
normalMax = 255
beruMin = 0
beruMax = 1

#set up storage
trainImgsOri, testImgsOri, trainImgsTouched, testImgsTouched = [],[],[],[]

#convert to matrix
arrayToMatrix(trainImgs, trainImgsOri, oriDim)
arrayToMatrix(testImgs, testImgsOri, oriDim)

#thresholding matrix
thresholding(trainImgsOri, threshold, beruMin, beruMax)
thresholding(testImgsOri, threshold, beruMin, beruMax)

#shrink image and then stretch image
shrinkAndResize(trainImgsOri, trainImgsTouched, stretchDim)
shrinkAndResize(testImgsOri, testImgsTouched ,stretchDim)

#Add training and testing code
	


