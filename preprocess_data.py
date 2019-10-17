import numpy as np
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize 


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

sentiment_tweet_dataframe = pandas.read_csv('sentiment-tweet-data.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
sentiment_tweet_dataframe = sentiment_tweet_dataframe.reindex(np.random.permutation(sentiment_tweet_dataframe.index))

def preprocess_text(text):
    ''' Preprocesses text by removing special characters, removing urls, lowercasing text, 
        removing stop words, and stemming the rest
            Args:
                text: string of the a tweets text
            Returns:
                A string of the text with the special characters and urls removed, loswercased text, 
                stopwords removed, and stemming of words
    '''
    text = re.sub(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text.lower()).strip()
    new_text = [stemmer.stem(token) for token in word_tokenize(text) if token not in stop_words and len(token) > 1]
    return ' '.join(new_text)

def preprocess_value(value):
    ''' Preprocesses values by making sure they are either 0 or 1 
            Args:
                value: integer that is either 0 or 4
            Returns:
                The given value or 1 if the value is 4
    '''
    if value == 4: return 1
    else: return value

def preprocess_data(sentiment_tweet_dataframe):
    ''' Prepares features from sentiment_tweet_data for model use.
            Args: 
                sentiment_tweet_data: A pandas dataframe of sentiment140 twitter data from kaggle
            Returns:
                A dataframe with features to be used for the model.
    '''
    selected_features = sentiment_tweet_dataframe[['target', 'text']]
    processed_data = selected_features.copy()
    # Remove links and secial characters from the lowercased text
    processed_data['text'] = processed_data['text'].apply(lambda x: preprocess_text(x))
    processed_data = processed_data.dropna()
    processed_data['target'] = processed_data['target'].apply(lambda x: preprocess_value(x))
    return processed_data

# Split data into training, validation, and test data
np.random.seed(42)
training_data, validation_data, test_data = np.split(sentiment_tweet_dataframe, \
    [int(.6*len(sentiment_tweet_dataframe)), int(.8*len(sentiment_tweet_dataframe))])

# Get processed data
training_data = preprocess_data(training_data)
validation_data = preprocess_data(validation_data)
test_data = preprocess_data(test_data)

# Save processed data to file for use in model training
training_data.to_csv('./data/training_data.csv')
validation_data.to_csv('./data/validation_data.csv')
test_data.to_csv('./data/test_data.csv')
