from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

sentiment_tweet_data = pandas.read_csv('sentiment-tweet-data.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
print(len(sentiment_tweet_data))
print(sentiment_tweet_data.head())