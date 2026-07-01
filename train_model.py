#Leonard Dixon pset 1

#import modules and libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

#load data
imdbDataset = pd.read_csv("IMDB Dataset.csv")

print("\nChecking if data loaded properly by looking at the head\n",imdbDataset.head())



#load data into variables
X = imdbDataset["review"]
y = imdbDataset["sentiment"]

print("\n Finished created X and y\n")


#create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

print("\n Finished creating pipeline\n")

#train model on pipeline
pipeline.fit(X, y)

print("\n Finished training model\n")

#save model to drive
joblib.dump(pipeline, 'sentiment_model.pkl')

print("\n Finished saving model\n")
