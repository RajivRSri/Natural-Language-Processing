#Set Working Directory
setwd("C:/R/Sentiment Analysis")
list.files()

#Read File
data = read.csv("tweets.csv", header = TRUE)

#Check Loaded Data
str(data)
glimpse(data)
head(data)
names(data)
dim(data)
summary(data)

#Build Corpus
library(tm)
corpus = iconv(data$text)
corpus = Corpus(VectorSource(corpus))
inspect(corpus)

#Clean Data
corpus = tm_map(corpus, tolower)
inspect(corpus[1:5])

corpus = tm_map(corpus, removePunctuation)
inspect(corpus[1:5])

corpus = tm_map(corpus, removeNumbers)
inspet(corpus[1:5])

corpus = tm_map(corpus, removeWords, stopwords("english"))
inspect(corpus[1:5])                

removeURL = function(x) gsub('http[[:alnum:]]', '', x)
corpus = tm_map(corpus, content_transformer(removeURL))
inspect(corpus[1:5])

corpus = tm_map(corpus, stripWhitespace)
inspect(corpus[1:5])

corpus = tm_map(corpus, stemDocument)
inspect(corpus[1:5])

corpus = tm_map(corpus, removeWords, c("aapl"))
inspect(corpus[1:5])

#Create Term Document Matrix
tdm = TermDocumentMatrix(corpus)
tdm = as.matrix(tdm)
tdm[1:10, 1:20]

#Bar Plot
w = rowSums(tdm)
w = subset(w,
           w >= 25)
barplot(w,
        las = 2,
        col = rainbow(50))
#WordCloud
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
w = sort(rowSums(tdm), decreasing = TRUE)
set.seed(222)
w = data.frame(names(w), w)
wordcloud2(w,
           size = 0.7,
           shape = "circle",
           rotateRatio = 0.5,
           minSize = 1)

#Sentiment Analysis
library(syuzhet)
data = read.csv("apple.csv", header = TRUE)
corpus = iconv(data$text)
sentiment = get_nrc_sentiment(corpus)
head(sentiment)
corpus[1]
get_nrc_sentiment("@option_snipper")
barplot(colSums(sentiment),
        las = 2,
        col = rainbow(10),
        ylab = "Count",
        main = "Sentiment Scores for Apple Tweets")









