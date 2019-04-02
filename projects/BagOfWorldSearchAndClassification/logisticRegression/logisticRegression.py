import processData as pData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

#constants
file = 'yelp_2k.csv'
mostFreqWork = 'mostFreqWord.txt'
stopWords = ['who', 'has', 'then', 'her', 'more', 'said', 'got', 'did', 'which', 'by', 'after', 'even', 'only', 'do', 'been', 'us', 'your', 'what', 'can', 'or', 'them', 'about', 'go', 'will', 'here', 'their', 'just', 'an', 'up', 'back', 'she', 'our', 'get', 'service', 'would', 'out', 'when', 'all', 'if', 'he', 'as', 'there', 'be', 'are', 'were', 'at', 'me', 'had', 'but', 'have', 'on', 'with', 'you', 'we', 'this', 'they', 'that', 'is', 'my', 'in', 'for', 'of', 'it', 'was', 'to', 'and', 'the']
minThreshold, maxThreshold = 5, 0.1

#graph word distribution
def graphWordFreq(x,y):
	plt.scatter(x,y)
	plt.xlabel('word rank')
	plt.ylabel('word count')
	plt.show()

#train the bag of words
def getBagOfWords(documents, stopWords, minThreshold, maxThreshold):
	vectorizer = CountVectorizer()
	vectorizer.stop_words = stopWords
	vectorizer.min_df = minThreshold
	#vectorizer.max_df = maxThreshold
	X = vectorizer.fit_transform(documents)
	return vectorizer, X

#get documents and stars
documents, stars = pData.getDataWithoutStopingWord(file)

#get bag of words
vectorizer, wordVectors = getBagOfWords(documents, stopWords, minThreshold, maxThreshold)
featureCount = len(vectorizer.get_feature_names())
totalCount = np.sum(wordVectors.toarray(),0)
featureName = vectorizer.get_feature_names()

	
y = sorted(np.sum(wordVectors.toarray(),0), reverse = True)
x = [i for i in range(1,featureCount+1)]
graphWordFreq(x,y)


print(X.toarray())