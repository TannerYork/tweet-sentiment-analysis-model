from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd

training_dataframe = pd.read_csv('./data/training_data.csv',  encoding="ISO-8859-1", names=['Sentiment', 'SentimentText'])
training_dataframe = training_dataframe.dropna()

validation_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['Sentiment', 'SentimentText'])
validation_dataframe = validation_dataframe.dropna()

test_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['Sentiment', 'SentimentText'])
test_dataframe = test_dataframe.dropna()

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(40, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(training_dataframe['SentimentText'].values, training_dataframe['Sentiment'].values, 
                    validation_data=(validation_dataframe['SentimentText'].values, validation_dataframe['Sentiment'].values), 
                    batch_size=512, epochs=100, verbose=1)
results = model.evaluate(test_dataframe['SentimentText'].values, test_dataframe['Sentiment'].values, verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
model.save('tweet-sentiment.h5')