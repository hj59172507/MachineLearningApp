import processData as pData
from sklearn.feature_extraction.text import CountVectorizer

#get documents and stars
file = 'yelp_2k.csv'
documents, stars = pData.getDataWithoutStopingWord(file)

