"""
__file__

    preprocessing.py

__description__

    Preprocess data in different ways.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import logging
import os
import pandas as pd
import pickle
import re
from nltk.stem.snowball import PorterStemmer

from globals import CONFIG

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, CONFIG['PREPROCESS_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']


def save_preprocessed_data(data, name):

    if not os.path.exists(PREPROCESS_DIR):
        os.makedirs(PREPROCESS_DIR)

    path = os.path.join(PREPROCESS_DIR, name + '.pickle')
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_preprocessed_data(name):

    path = os.path.join(PREPROCESS_DIR, name + '.pickle')
    if not os.path.exists(path):
        raise ValueError('File %s does not exists.' % path)

    return pickle.load(open(path, 'rb'))


def check_if_preprocessed_data_exists(name):
    return os.path.exists(os.path.join(PREPROCESS_DIR, name + '.pickle'))


def create_words(str_, regex=r'\W+'):

    new_str = re.sub(regex, ' ', str_.lower())
    return new_str.split(' ')


def preprocess_data():

    logging.info('PREPROCESSING DATA')

    # Read data.
    df_train = pd.read_csv(TRAIN_FILE, nrows=TRAIN_NROWS)
    df_test = pd.read_csv(TEST_FILE, nrows=TEST_NROWS)
    df_test.rename(columns={'test_id': 'id'}, inplace=True)

    # Merge data together.
    wanted_cols = ['id', 'question1', 'question2']
    data = pd.concat([df_train[wanted_cols + ['is_duplicate']],
                      df_test[wanted_cols]], ignore_index=True)

    # Create standard preprocessing: split data by non-alphanumerical ch,
    # lower case.
    name = 'standard_preprocess'
    if not check_if_preprocessed_data_exists(name):
        data_preprocessed = pd.DataFrame(data['id'])
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: str(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: str(x))
        data_preprocessed['words1'] = data_preprocessed['question1'].apply(
            lambda x: create_words(x))
        data_preprocessed['words2'] = data_preprocessed['question2'].apply(
            lambda x: create_words(x))
        save_preprocessed_data(data_preprocessed, name)

    name = 'stemma_preprocess'
    if not check_if_preprocessed_data_exists(name):
        # Load standard preprocessed data.
        data_preprocessed = load_preprocessed_data('standard_preprocess')

        # Stemmatize words.
        stemmer = PorterStemmer(ignore_stopwords=False)
        data_preprocessed['words1'] = data_preprocessed['words1'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['words2'] = data_preprocessed['words2'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['question1'] = data_preprocessed['words1'].apply(
            lambda x: ' '.join(x))
        data_preprocessed['question2'] = data_preprocessed['words2'].apply(
            lambda x: ' '.join(x))
        save_preprocessed_data(data_preprocessed, name)

    logging.info('DATA PREPROCESSED')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    preprocess_data()
