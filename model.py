from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
# from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_extraction.text import CountVectorizer

training_dataframe = pd.read_csv('./data/training_data.csv',  encoding="ISO-8859-1", names=['target', 'text'])
training_dataframe = training_dataframe.dropna()
targets = training_dataframe.pop('target')
training_dataset = tf.data.Dataset.from_tensor_slices((training_dataframe.values, targets.values))

validation_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['target', 'text'])
validation_dataframe = validation_dataframe.dropna()
targets = validation_dataframe.pop('target')
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_dataframe.values, targets.values))

test_dataframe = pd.read_csv('./data/training_data.csv', encoding="ISO-8859-1", names=['target', 'text'])
test_dataframe = test_dataframe.dropna()
targets = test_dataframe.pop('target')
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataframe.values, targets.values))

# vectorizor = CountVectorizer().fit(training_dataframe)
# training_vectorized = training_dataframe.apply(vectorizor.transform)
# gnb = GaussianNB()
# print(training_vectorized[:5])
# model = gnb.fit(training_vectorized.values, targets.values)

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
history = model.fit(training_dataset, epochs=20, validation_data=validation_dataset, verbose=1)
results = model.evaluate(test_dataset, verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))