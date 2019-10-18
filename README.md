# Twitter-Sentiment-Analysis

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```pip install -r requirements.txt --no-index --find-links file:///tmp/packages```

### Installing

There are two ways you can train/modify the model.

The first way is in a jupyter notebook (recomended)
1. open jupyter notebook
2. Run the code blocks in order
3. Modefy model and train in the training code block

The second way is training in python
1. Install the twitter sentemient dataset found [here](https://www.kaggle.com/kazanova/sentiment140)
2. Run the preprocess_data file
```python3 preprocess_data.py```
3. Modify model and run the model file
```python3 model.py```


## Built With

* [Tensorflow](https://www.tensorflow.org)
* [Pandas](https://pandas.pydata.org)
* [NLTK](https://www.nltk.org)


## Authors

* **Tanner York**
