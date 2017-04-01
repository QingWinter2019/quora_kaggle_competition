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
import logging
import re

from feature_classification import add_features, dump_feature_classes_and_dict
from globals import CONFIG
from pickle_utils import dump_features, check_if_exists

from tfidf_features import create_tfidf_features
from tfidf_svd_features import create_svd_tfidf_features, create_raw_tfidf_features
from tfidf_svd_features import create_common_vocabulary_svd_tfidf_features
from tfidf_svd_features import create_common_vocabulary_raw_tfidf_features

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']


def create_words(str_, regex=r'\W+'):

    new_str = re.sub(regex, ' ', str(str_).lower())
    return new_str.split(' ')


def feature_engineering():

    logging.info('FEATURE ENGINEERING')
    # Read data.
    df_train = pd.read_csv(TRAIN_FILE, nrows=TRAIN_NROWS)
    df_test = pd.read_csv(TEST_FILE, nrows=TEST_NROWS)
    df_test.rename(columns={'test_id': 'id'}, inplace=True)

    # Merge data together.
    wanted_cols = ['id', 'question1', 'question2']
    data = pd.concat([df_train[wanted_cols + ['is_duplicate']],
                      df_test[wanted_cols]])

    # Preprocess data.
    data['question1'] = data['question1'].apply(lambda x: str(x))
    data['question2'] = data['question2'].apply(lambda x: str(x))
    data['words1'] = data['question1'].apply(lambda x: create_words(x))
    data['words2'] = data['question2'].apply(lambda x: create_words(x))

    # Create features.
    create_common_words_count_features(data)
    create_tfidf_features(data, columns=['question2'], qcol='question1', unique=False)
    create_raw_tfidf_features(data, columns=['question1', 'question2'])
    create_svd_tfidf_features(columns=['question1', 'question2'])
    create_common_vocabulary_raw_tfidf_features(data, 'question1', 'question2')
    create_common_vocabulary_svd_tfidf_features()

    dump_feature_classes_and_dict()
    logging.info('FINISHED FEATURE ENGINEERING')


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


def create_common_words_count_features(data):

    logging.info('Creating common words features')
    feature_class = 'common_words'
    if check_if_exists(feature_class):
        logging.info('Common words features already created')
        return

    res = pd.DataFrame()
    res['common_words'] = data.apply(
        lambda x: common_words_count(x['words1'], x['words2']), axis=1)
    res['len1'] = data['words1'].apply(lambda x: len(x))
    res['len2'] = data['words2'].apply(lambda x: len(x))
    res['lenunion'] = data.apply(
        lambda x: union_words_count(x['words1'], x['words2']), axis=1)
    res['distance1'] = res['common_words'] / res['lenunion']
    res['distance2'] = res['common_words'] / (res['len1'] + res['len2'])

    res['common_words_len'] = data.apply(
        lambda x: common_words_len(x['words1'], x['words2']), axis=1)
    res['abs_len1'] = data['words1'].apply(lambda x: words_len(x))
    res['abs_len2'] = data['words2'].apply(lambda x: words_len(x))
    res['abs_lenunion'] = data.apply(
        lambda x: union_words_len(x['words1'], x['words2']), axis=1)
    res['absdistance1'] = res['common_words_len'] / res['abs_lenunion']
    res['absdistance2'] = res['common_words_len'] / (res['abs_len1'] + res['abs_len2'])

    features = res.columns.tolist()
    add_features(feature_class, features)
    dump_features(feature_class, res)
    logging.info('Common words features are created and saved to pickle file.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    feature_engineering()
