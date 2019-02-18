from scipy.spatial.distance import pdist
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import numpy as np
from numpy.linalg import svd

#constants
pComponents, categoryCount = 20, 10
files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

#to read data, script reference to https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

#put data into appropriate datalist according to data label
def putIntoCategory(dataList, data):
	arr = data[b'data']
	labels = data[b'labels']
	for labelIndex in range(len(labels)):
		dataList[labels[labelIndex]].append(arr[labelIndex])

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
		meanlist.append(np.mean(category, 0))
	return meanList

dataList = readAll(files)
meanList = getMeanForAll(dataList)

print(dataList)




