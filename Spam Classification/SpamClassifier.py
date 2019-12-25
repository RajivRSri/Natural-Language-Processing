#Set Working Directory
import os
os.chdir ("C:/R")

#Load Dataset
import pandas as pd
data = pd.read_csv("SpamClassifier/smsspamcollection/SMSSpamCollection",
                   sep = "\t",
                   names = ["label", "text"])
#Check Loaded Data
data.head()
data.info()
data.shape
data.isnull().values.any()

#Exploratory Data Analysis
data.label.value_counts()
data["label"].value_counts()

import matplotlib.pyplot as plt
count_label = pd.value_counts(data.label, sort = True)
count_label.plot(kind = "bar", rot = 0)
plt.title("Class Distribution")
plt.xticks(range(2), ["ham", "spam"])
plt.xlabel("Class")
plt.ylabel("Frequency")

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

numRows = data.shape[0]
corpus = []
lemmatizer = WordNetLemmatizer()
for index in range (numRows):
    review = re.sub('[^a-zA-Z]', ' ', data["text"][index])
    review = review.upper()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

#Building Bag Of Words Model
from sklearn.feature_extraction.text import TfidfVectorizer
Tf = TfidfVectorizer()
X = Tf.fit_transform(corpus).toarray()
    
#One Hot Encoding
Y = pd.get_dummies(data["label"])
Y = Y.iloc[:,1]

#Apply Oversampling
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
X_res, Y_res = smk.fit_sample(X, Y)
print(X_res.shape, Y_res.shape)

#Split into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 0)

#Build Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#Create confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(Y_test, Y_pred)
print(conf_mat)

#Check Model Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print("Model Accuracy = {}".format(accuracy))
