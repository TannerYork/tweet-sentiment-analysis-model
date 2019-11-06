# Twitter-Sentiment-Analysis

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```pip install -r requirements.txt --no-index --find-links file:///tmp/packages```

### Installing

There are two ways you can train/modify the model. The data is already preprocessed for you, so no need to download the dataset unless you want to. The sentiment data can be found [here](https://www.kaggle.com/kazanova/sentiment140) if your interested.

The first way is in a jupyter notebook <br>
(currently having trouble connecting notebooks to virtualenv. If you want to use this method, you'll need to download the requirments outside of a virtualenv or set the connection yourself.)

1. open jupyter notebook
2. Run the code blocks in order
3. Modefy model and train in the training code block

The second way is training in python

1. Modify model and run the model file
```python3 model.py```

## Get the Model
If you want to just use the model with python, follow these steps

1. Download tweet_sa_model.h5
2. Open new or existing project in code editor
3. Downlod tenforslow and tensorflow_hub
```
pip3 install tensorflow
pip3 install tensorflow_hub
```
4. Import tensorflow keras and tensorflow_hub
```
from tensorflow import keras
import tensorflow_hub as hub
```
5. Load model with hub KerasLayer<br>
```
model = keras.models.load_model('./twitter_sa_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
```
6. Make perdictions from preprocessed data<br>
```
model.predict(['hating summer classes loving new apt complete bamfin deck'])
```



## Built With

* [Tensorflow](https://www.tensorflow.org)
* [Pandas](https://pandas.pydata.org)
* [NLTK](https://www.nltk.org)


## Authors

* **Tanner York**

