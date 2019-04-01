import processData as pData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

#graph word distribution
def graphWordFreq(x,y):
	plt.scatter(x,y)
	plt.xlabel('word rank')
	plt.ylabel('word count')
	plt.show()

#get documents and stars
file = 'yelp_2k.csv'
documents, stars = pData.getDataWithoutStopingWord(file)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
featureCount = len(vectorizer.get_feature_names())

y = sorted(np.sum(X.toarray(),0), reverse = True)
x = [i for i in range(1,featureCount+1)]
graphWordFreq(x,y)


print(X.toarray())