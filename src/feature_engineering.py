"""
__file__

    feature_classification.py

__description__

    Create all features.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import os
import pandas as pd
import pickle
import re

from feature_classification import add_features, dump_features
from globals import CONFIG

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
TRAIN_PREPROCESS_FILE = os.path.join(DATA_DIR, 'train_preprocess.csv')
TEST_PREPROCESS_FILE = os.path.join(DATA_DIR, 'test_preprocess.csv')


def create_words(str_):

    # print(str_)
    # Replace all non-alphanumberic characters with a space.
    new_str = re.sub(r'\W+', ' ', str(str_).lower())
    return new_str.split(' ')


def feature_engineering():

    # Read data.
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_test.rename(columns={'test_id': 'id'}, inplace=True)

    # Merge data together.
    wanted_cols = ['id', 'question1', 'question2']
    data = pd.concat([df_train[wanted_cols + ['is_duplicate']],
                      df_test[wanted_cols]])

    #
    data['words1'] = data['question1'].apply(lambda x: create_words(x))
    data['words2'] = data['question2'].apply(lambda x: create_words(x))

    # Add features.
    data = add_common_words_count_features(data)
    dump_features()

    # Split data back into train and test dataset and pickle.
    data[:len(df_train)].to_csv(TRAIN_PREPROCESS_FILE, index=False)
    data[len(df_train):].to_csv(TEST_PREPROCESS_FILE, index=False)


def words_len(words):
    return sum([len(word) for word in words])


def common_words_count(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return len(set1.intersection(set2))


def common_words_len(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return words_len(set1.intersection(set2))


def union_words_count(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return len(set1.union(set2))


def union_words_len(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return words_len(set1.union(set2))


def add_common_words_count_features(data, inplace=True):

    data['common_words'] = data.apply(
        lambda x: common_words_count(x['words1'], x['words2']), axis=1)
    data['len1'] = data['words1'].apply(lambda x: len(x))
    data['len2'] = data['words2'].apply(lambda x: len(x))
    data['lenunion'] = data.apply(
        lambda x: union_words_count(x['words1'], x['words2']), axis=1)
    data['distance1'] = data['common_words'] / data['lenunion']

    data['common_words_len'] = data.apply(
        lambda x: common_words_len(x['words1'], x['words2']), axis=1)
    data['abs_len1'] = data['words1'].apply(lambda x: words_len(x))
    data['abs_len2'] = data['words2'].apply(lambda x: words_len(x))
    data['abs_lenunion'] = data.apply(
        lambda x: union_words_len(x['words1'], x['words2']), axis=1)
    data['absdistance1'] = data['common_words_len'] / data['abs_lenunion']

    add_features('common_words', ['common_words', 'len1', 'len2', 'lenunion',
                                  'distance1', 'common_words_len', 'abs_len1',
                                  'abs_len2', 'abs_lenunion', 'absdistance1'])
    return data


if __name__ == '__main__':
    feature_engineering()
