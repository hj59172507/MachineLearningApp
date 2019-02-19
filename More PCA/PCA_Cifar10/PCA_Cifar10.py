from scipy.spatial.distance import pdist
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

#constants
nComp, categoryCount, mdsComp = 20, 10, 2
eucFile, pcaFile = 'partb_distances.csv','partc_distances.csv'
files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']   
category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#to read data, script reference to https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

#put data into appropriate datalist according to data label
def putIntoCategory(dataList, data):
	arrs = data[b'data'].astype(float)
	labels = data[b'labels']
	for labelIndex in range(len(labels)):
		dataList[labels[labelIndex]].append(arrs[labelIndex])

#read data from each file and put into appropriate category
def readAll(files):
	dataList = [ [] for i in range(categoryCount) ]
	for file in files:
		data = unpickle(file)
		putIntoCategory(dataList, data)
	return dataList

#find mean of each category
def getMeanForAll(dataList):
	meanList = []
	for category in dataList:
		meanList.append(np.mean(category, 0))
	return meanList

#reconstruct a category using pca and return square different between reconstructed and original version
def pcaRecon(pca, data, mean):
	temp = 0
	for d in data:
		d = d.reshape(1,-1)
		recon = np.dot(pca.transform(d)[:,:nComp], pca.components_[:nComp,:]) + mean
		temp += np.square(sum(euclidean_distances(recon.reshape(1,-1), d)))[0]
	return temp / len(data)

#reconstruct data using svd with specified principle componenets and return average square differences between reconstructed and original version
def getAvgSqDiff(dataList, meanList):
	diffList = []
	for i in range(len(dataList)):
		pca = PCA(n_components=nComp)
		pca.fit(dataList[i])
		diffList.append(pcaRecon(pca, dataList[i], meanList[i]))
	return diffList

#plot the bar graph for mean different with each category
def plotBar(List):
	yPos = np.arange(len(category))
	plt.bar(yPos, List, align='center')
	plt.ylabel('mean difference')
	plt.xticks(yPos, category, rotation = 90)
	plt.title('Mean difference with each Class')
	plt.show()

#plot the mean image of each class
def plotMeanImage(meanList):
	for x in range(len(meanList)):
		mean = meanList[x]
		img = []
		for i in range(32):
			img.append([])
			for j in range(32):
				img[i].append([])
				img[i][j] = np.array([mean[i*32+j],mean[i*32+j+32*32],mean[i*32+j+32*32*2]]).astype(np.uint8)
		plt.imshow(img)
		plt.title('Mean image of Category '+str(x) + ' : ' + category[x])
		plt.show()

dataList = readAll(files)
meanList = getMeanForAll(dataList)

#Compute mean iamge and sum square difference
#plotMeanImage(meanList)
#diffList = getAvgSqDiff(dataList, meanList)
#plotBar(diffList)

#compute euclidean distance between all pair of class using mean image
def getDisMatrixEuclidean(meanList):
	eucDisMat = []
	for i in range(len(meanList)):
		eucDisMat.append([])
		for j in range(len(meanList)):
			dis = sum(euclidean_distances(meanList[i].reshape(1,-1), meanList[j].reshape(1,-1)))[0]
			eucDisMat[i].append(dis*dis)
	return eucDisMat

#compute distance using PCA, distance of A -> B denoted by E(A->B) is define as (E(A|B) + E(B|A))/2 
#where E(A|B) is computed by using the mean of A and first 20 principal components from B to reconstruct A
def getDisMatrixPCA(meanList, dataList):
	pcaDisMat = []
	for i in range(len(meanList)):
		pcaDisMat.append([])
		for j in range(len(meanList)):
			pcaDisMat[i].append([])
	for i in range(len(meanList)):
		for j in range(i,len(meanList)):
			if(i==j):
				pca = PCA(n_components=nComp)
				pca.fit(dataList[i])
				distance = pcaRecon(pca, dataList[i], meanList[i])
				pcaDisMat[i][j] = distance
			else:
				pcaI = PCA(n_components=nComp)
				pcaJ = PCA(n_components=nComp)
				pcaI.fit(dataList[i])
				pcaJ.fit(dataList[j])
				pcaI.mean_ = meanList[j]
				pcaJ.mean_ = meanList[i]
				tempI = pcaRecon(pcaJ, dataList[i], meanList[i])
				tempJ = pcaRecon(pcaI, dataList[j], meanList[j])
				distance = (tempI + tempJ)/2
				pcaDisMat[i][j] = distance
				pcaDisMat[j][i] = distance
	return pcaDisMat

#helper to write result to a file
def writeAsCSVTo(file, data):
	with open(file, 'w') as file:
		fwriter = csv.writer(file,delimiter=',')
		for row in data:
			fwriter.writerow(row)

#multi-dimensional scaling using distance matrix
def MDS(disMat):
	N = len(disMat)
	A = np.identity(N)
	W = -0.5 * np.dot(A, np.dot(disMat, A.T))
	lam, U = np.linalg.eig(W)
	lam = np.diag(np.sqrt(abs(lam)))[:mdsComp,:mdsComp]
	U = U[:,:mdsComp]
	Y = np.dot(U,lam)
	x, y = [],[]
	for point in Y:
		x.append(point[0])
		y.append(point[1])
	return (x,y)

#plot a scatter plot with given title
def scatterPlot(x, y, title):
	for i in range(len(category)):
		plt.scatter(x[i], y[i])
		plt.text(x[i], y[i], category[i])
	plt.xlabel("First component")
	plt.ylabel("Second component")
	plt.title(title)
	plt.show()

#Compute distance martrix, perform multi-dimensional scaling and plot the first two component
eucDisMat = getDisMatrixEuclidean(meanList)
writeAsCSVTo(eucFile, eucDisMat)
x,y = MDS(eucDisMat)
scatterPlot(x, y, "MDS using Euclidean distance")

#Compute distance martrix define by E(A->B) = (E(A|B)+E(B|A))/2, perform multi-dimensional scaling and plot the first two component
pcaDisMat = getDisMatrixPCA(meanList, dataList)
writeAsCSVTo(pcaFile, pcaDisMat)
x,y = MDS(pcaDisMat)
scatterPlot(x, y, "MDS using pca reconstruct distance")




