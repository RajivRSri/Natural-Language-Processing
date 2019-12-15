#Set Working Directory
import os
os.chdir("C:/R")

#Load Dataset
import pandas as pd
data = pd.read_csv("customer_reviews.csv")

#Check Loaded data
data.head()
data.info()
data.columns
data.shape

#EDA
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
numRows = data.shape[0]
corpus = []

for index in range(numRows):
    review = re.sub('[^a-zA-Z]', ' ', data["text"][index])
    review = review.upper()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
#Tfidf Model
from sklearn.feature_extraction.text import TfidfVectorizer
Tf = TfidfVectorizer().fit_transform(corpus).toarray()

#Analysing Sentiments
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()
data["Score"] = data["text"].apply(lambda X: sentiment.polarity_scores(X))
data["Compound_Score"] = data["Score"].apply(lambda X: X["compound"])
data["Pos_Neg"] = data["Compound_Score"].apply(lambda X: np.where(X > 0, "Positive", "Negative"))
data["Pos_Neg"].value_counts()
