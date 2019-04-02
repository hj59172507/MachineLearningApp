import processData as pData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

#constants
file = 'yelp_2k.csv'
mostFreqWork = 'mostFreqWord.txt'
stopWords = ['who', 'has', 'then', 'her', 'more', 'said', 'got', 'did', 'which', 'by', 'after', 'even', 'only', 'do', 'been', 'us', 'your', 'what', 'can', 'or', 'them', 'about', 'go', 'will', 'here', 'their', 'just', 'an', 'up', 'back', 'she', 'our', 'get', 'service', 'would', 'out', 'when', 'all', 'if', 'he', 'as', 'there', 'be', 'are', 'were', 'at', 'me', 'had', 'but', 'have', 'on', 'with', 'you', 'we', 'this', 'they', 'that', 'is', 'my', 'in', 'for', 'of', 'it', 'was', 'to', 'and', 'the']
minThreshold, maxThreshold = 5, 0.1
matchingReview = ['horrible customer service']
neighbors = 5

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

#get documents and stars
documents, stars = pData.getDataWithoutStopingWord(file)

#get bag of words
vectorizer, wordVectors = getBagOfWords(documents, stopWords, minThreshold, maxThreshold)
#featureCount = len(vectorizer.get_feature_names())
#totalCount = np.sum(wordVectors.toarray(),0)
#featureName = vectorizer.get_feature_names()

#print the word distribution
#y = sorted(np.sum(wordVectors.toarray(),0), reverse = True)
#x = [i for i in range(1,featureCount+1)]
#graphWordFreq(x,y)

#using cos-distance to pick 5 reviews matching 'Horrible customer service'
reviewVec = vectorizer.transform(matchingReview).toarray()
nnModel = getNNModel(wordVectors)
bestNeighbors = nnModel.kneighbors(reviewVec, neighbors)

with open('text.txt', 'w', encoding='utf-8') as f:
	for i in bestNeighbors[1][0]:
		f.write(documents[i]+'\n')

#graph all cos distance
#allNeighbors = nnModel.kneighbors(reviewVec, len(wordVectors))
#y = sorted(allNeighbors[0][0], reverse = True)
#x = [i for i in range(1,len(wordVectors)+1)]
#graphWordFreq(x,y,'rank','cosine distance')


print(X.toarray())