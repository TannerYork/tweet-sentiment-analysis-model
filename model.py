from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd

training_dataframe = pd.read_csv('./data/training_data.csv',  encoding="ISO-8859-1", names=['target', 'text'])
training_dataframe = training_dataframe.dropna()

validation_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['target', 'text'])
validation_dataframe = validation_dataframe.dropna()

test_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['target', 'text'])
test_dataframe = test_dataframe.dropna()

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=SGD(ls=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(training_dataframe['text'].values, training_dataframe['target'].values, 
                    validation_data=(validation_dataframe['text'].values, validation_dataframe['target'].values), 
                    batch_size=512, epochs=20, verbose=1)
results = model.evaluate(test_dataframe['text'].values, test_dataframe['target'].values, verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))