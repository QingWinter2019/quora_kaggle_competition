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
from preprocessing import load_preprocessed_data

from tfidf_features import create_tfidf_features
from tfidf_svd_features import create_svd_tfidf_features
from tfidf_svd_features import create_raw_tfidf_features
from tfidf_svd_features import create_common_vocabulary_svd_tfidf_features
from tfidf_svd_features import create_common_vocabulary_raw_tfidf_features
from tfidf_svd_features import create_distance_tfidf_features
from grouping_features import create_grouping_features
from word2vec_features import create_word2vec_features
from logistic_features import create_logistic_features

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

    # Create standard features.
    data = load_preprocessed_data('standard_preprocess')
    create_common_words_count_features(data)
    create_tfidf_features(data, columns=['question2'], qcol='question1',
                          unique=False)
    create_raw_tfidf_features(data, columns=['question1', 'question2'])
    create_svd_tfidf_features(columns=['question1', 'question2'])
    create_common_vocabulary_raw_tfidf_features(data, 'question1', 'question2')
    create_common_vocabulary_svd_tfidf_features()
    create_distance_tfidf_features('question1', 'question2')
    create_word2vec_features(data, 'words1', 'words2')
    create_grouping_features(data)

    # Create stemma features.
    data = load_preprocessed_data('stemma_preprocess')
    create_common_words_count_features(data, pref='stemma_')
    create_tfidf_features(data, columns=['question2'], qcol='question1',
                          unique=False, pref='stemma_')
    create_raw_tfidf_features(data, columns=['question1', 'question2'],
                              pref='stemma_')
    create_svd_tfidf_features(columns=['question1', 'question2'],
                              pref='stemma_')
    create_common_vocabulary_raw_tfidf_features(data, 'question1', 'question2',
                                                pref='stemma_')
    create_common_vocabulary_svd_tfidf_features(pref='stemma_')
    create_distance_tfidf_features('question1', 'question2', pref='stemma_')
    create_word2vec_features(data, 'words1', 'words2', pref='stemma_')
    create_grouping_features(data, pref='stemma_')

    # Create stemma stopwords features.
    data = load_preprocessed_data('stemma_preprocess_stopwords')
    create_common_words_count_features(data, pref='stemma_stopwords_')
    create_tfidf_features(data, columns=['question2'], qcol='question1',
                          unique=False, pref='stemma_stopwords_')
    create_raw_tfidf_features(data, columns=['question1', 'question2'],
                              pref='stemma_stopwords_')
    create_svd_tfidf_features(columns=['question1', 'question2'],
                              pref='stemma_stopwords_')
    create_common_vocabulary_raw_tfidf_features(data, 'question1', 'question2',
                                                pref='stemma_stopwords_')
    create_common_vocabulary_svd_tfidf_features(pref='stemma_stopwords_')
    create_distance_tfidf_features('question1', 'question2', pref='stemma_stopwords_')
    create_word2vec_features(data, 'words1', 'words2', pref='stemma_stopwords_')
    create_grouping_features(data, pref='stemma_stopwords_')

    # Create Damerau Levenshtein features.
    # For now we create not all features to see if there is at all benefit.
    data = load_preprocessed_data('dl_preprocess')
    create_common_words_count_features(data, pref='dl_')
    create_tfidf_features(data, columns=['question2'], qcol='question1',
                          unique=False, pref='dl_')
    create_grouping_features(data, pref='dl_')

    # Create Concat features.
    # For now we create only common_words_count and grouping features.
    data = load_preprocessed_data('concat_preprocess')
    create_common_words_count_features(data, pref='concat_')
    create_grouping_features(data, pref='concat_')

    # Create Noun features.
    # For now we create only common_words_count and grouping features.
    data = load_preprocessed_data('noun_preprocess')
    create_common_words_count_features(data, pref='noun_')
    create_grouping_features(data, pref='noun_')

    # Metafeatures.
    create_logistic_features()

    # Create clean_concat features.
    data = load_preprocessed_data('clean_concat')
    create_common_words_count_features(data, pref='clean_concat_')
    create_tfidf_features(data, columns=['question2'], qcol='question1',
                          unique=False, pref='clean_concat_')
    create_raw_tfidf_features(data, columns=['question1', 'question2'],
                              pref='clean_concat_')
    create_svd_tfidf_features(columns=['question1', 'question2'],
                              pref='clean_concat_')
    create_common_vocabulary_raw_tfidf_features(data, 'question1', 'question2',
                                                pref='clean_concat_')
    create_common_vocabulary_svd_tfidf_features(pref='clean_concat_')
    create_distance_tfidf_features('question1', 'question2', pref='clean_concat_')
    create_grouping_features(data, pref='clean_concat_')

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


def create_common_words_count_features(data, pref=''):

    logging.info('Creating common words features')
    feature_class = pref + 'common_words'
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
    res['absdistance2'] = res['common_words_len'] / (res['abs_len1'] +
                                                     res['abs_len2'])

    features = res.columns.tolist()
    add_features(feature_class, features)
    dump_features(feature_class, res)
    logging.info('Common words features are created and saved to pickle file.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    feature_engineering()
