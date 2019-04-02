import processData as pData
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

#constants
file = 'yelp_2k.csv'
mostFreqWork = 'mostFreqWord.txt'
stopWords = ['who', 'has', 'then', 'her', 'more', 'said', 'got', 'did', 'which', 'by', 'after', 'even', 'only', 'do', 'been', 'us', 'your', 'what', 'can', 'or', 'them', 'about', 'go', 'will', 'here', 'their', 'just', 'an', 'up', 'back', 'she', 'our', 'get', 'service', 'would', 'out', 'when', 'all', 'if', 'he', 'as', 'there', 'be', 'are', 'were', 'at', 'me', 'had', 'but', 'have', 'on', 'with', 'you', 'we', 'this', 'they', 'that', 'is', 'my', 'in', 'for', 'of', 'it', 'was', 'to', 'and', 'the']
minThreshold, maxThreshold = 5, 0.1
matchingReview = ['horrible customer service']
neighbors = 5
testProb = 0.1

#graph word distribution
def graphWordFreq(x,y,xlabel = 'word rank', ylabel = 'word count'):
	plt.scatter(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

#get the bag of words
def getBagOfWords(documents, stopWords, minThreshold, maxThreshold):
	vectorizer = CountVectorizer()
	vectorizer.stop_words = stopWords
	vectorizer.min_df = minThreshold
	vectorizer.max_df = maxThreshold
	X = vectorizer.fit_transform(documents)
	return vectorizer, X.toarray()

def getNNModel(wordVectors):
	neigh = NearestNeighbors()
	neigh.metric = 'cosine'
	neigh.fit(wordVectors)  
	return neigh

def plotScoreHisto(logitModel, data):
	one, five, prob = [],[], logitModel.predict_proba(data)
	for p in prob:
		if p[1] <= 0.5:
			one.append(p[1])
		else:
			five.append(p[1])
	plt.hist(one,30, alpha=0.5, label='1 star')
	plt.hist(five,30,alpha=0.5, label='5 stars' )
	plt.xlabel('predicted score')
	plt.ylabel('count')
	plt.legend(loc='north')
	plt.show()
	return

def getAccwith(logitModel, threshold, data, label):
	correct, prob = 0, logitModel.predict_proba(data)
	for i in range(len(prob)):
		if prob[i][1] > threshold and label[i] == 5:
			correct += 1
		if prob[i][1] <= threshold and label[i] == 1:
			correct += 1
	return correct / len(label)

#plot ROC curve to help visualize best threshold
def plotRoc(logitModel, data, label):
	thresholds = np.linspace(0,1,100)
	prob = logitModel.predict_proba(data)
	fpr, tpr, size = [],[],len([i for i in label if i == 5])
	for t in thresholds:
		fprTemp, tprTemp = 0,0
		for i in range(len(prob)):
			if prob[i][1] > t:
				if label[i] == 5:
					tprTemp += 1
				else:
					fprTemp += 1
		fpr.append(fprTemp/size)
		tpr.append(tprTemp/size)

	#find the best threshold that minimize fpr while maximize tpr
	max = 0
	index = -1
	for i in range(len(thresholds)):
		temp = tpr[i] - fpr[i]
		if temp > max:
			max = temp
			index = i
	print(max)
	print(thresholds[index])

	plt.plot(fpr, tpr, marker='.')
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.ylabel('true positive rate')
	plt.xlabel('false positive rate')
	plt.title('ROC Curve')
	plt.show()
	return
	

#get documents and stars
documents, stars = pData.getDataWithoutStopingWord(file)

#get bag of words
vectorizer, wordVectors = getBagOfWords(documents, stopWords, minThreshold, maxThreshold)
#featureCount = len(vectorizer.get_feature_names())
#totalCount = np.sum(wordVectors.toarray(),0)
#featureName = vectorizer.get_feature_names()

#print the word distribution
y = sorted(np.sum(wordVectors.toarray(),0), reverse = True)
x = [i for i in range(1,featureCount+1)]
graphWordFreq(x,y)

#using cos-distance to pick 5 reviews matching 'Horrible customer service'
reviewVec = vectorizer.transform(matchingReview).toarray()
nnModel = getNNModel(wordVectors)
bestNeighbors = nnModel.kneighbors(reviewVec, neighbors)

#graph all cos distance
allNeighbors = nnModel.kneighbors(reviewVec, len(wordVectors))
y = sorted(allNeighbors[0][0], reverse = True)
x = [i for i in range(1,len(wordVectors)+1)]
graphWordFreq(x,y,'rank','cosine distance')

#split data into test and train
test, testLabels, train, trainLabels = pData.splitData(wordVectors, testProb, stars)
logitModel = LogisticRegression().fit(train, trainLabels)

#get accuracy on training and testing data
#trainAcc = logitModel.score(train, trainLabels)
#testAcc = logitModel.score(test, testLabels)
plotScoreHisto(logitModel, train)

#get accuracy with different threshold
#testAcc = getAccwith(logitModel, 0.45, test, testLabels)
#trainAcc = getAccwith(logitModel, 0.45, train, trainLabels)

#plot roc curve
plotRoc(logitModel, test, testLabels)

print(X.toarray())