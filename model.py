from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas
import re


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

sentiment_tweet_dataframe = pandas.read_csv('sentiment-tweet-data.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
sentiment_tweet_dataframe = sentiment_tweet_dataframe.reindex(np.random.permutation(sentiment_tweet_dataframe.index))


def preprocess_features(sentiment_tweet_dataframe):
    ''' Prepares features from sentiment_tweet_data for model use.
            Args: 
                sentiment_tweet_data: A pandas dataframe of sentiment140 twitter data from kaggle
            Returns:
                A dataframe with features to be used for the model.
    '''
    selected_features = sentiment_tweet_dataframe['text']
    processed_features = selected_features.copy()
    # Remove links and secial characters from the lowercased text
    processed_features = processed_features.apply(lambda x: \
        re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', x.lower()).strip())
    return processed_features

def preprocess_targets(sentiment_tweet_dataframe):
    ''' Prepares targets feartures (labels) from sentiment_tweet_dataframe
            Args:
                sentiment_tweet_dataframe: a pandas dataframe of sentiment140 twitter data from kaggel
            Returns:
                A dataframe that contains the target feature
    ''' 
    output_targets = pandas.DataFrame()
    output_targets['positive'] = (sentiment_tweet_dataframe['target'] == 4)
    output_targets['negative'] = (sentiment_tweet_dataframe['target'] == 0)
    return output_targets

# Split data into training and validation 
TRAINING_AMOUNT = int(len(sentiment_tweet_dataframe)*0.80)
VALIDATION_AMOUNT = int(len(sentiment_tweet_dataframe)*0.20)

training_examples = preprocess_features(sentiment_tweet_dataframe.head(TRAINING_AMOUNT))
training_targets = preprocess_targets(sentiment_tweet_dataframe.head(TRAINING_AMOUNT))

validation_examples = preprocess_features(sentiment_tweet_dataframe.tail(VALIDATION_AMOUNT))
validation_targets = preprocess_targets(sentiment_tweet_dataframe.tail(VALIDATION_AMOUNT))

print("Training examples summary:")
print(training_examples.describe())
print("Validation examples summary:")
print(validation_examples.describe())

print("Training targets summary:")
print(training_targets.describe())
print("Validation targets summary:")
print(validation_targets.describe())