import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import pickle

from globals import CONFIG
from feature_classification import add_features
from pickle_utils import check_if_exists, dump_features

# Global Tfidf parameters.
MAX_DF = 0.75
MIN_DF = 10
NGRAM_RANGE = (1, 1)
N_COMPONENTS = 100
NORM = 'l1'
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])


def create_raw_tfidf_features(df_all, columns):

    logging.info('Creating raw tfidf features.')
    feature_class = 'raw_tfidf_%s' % columns[0]
    if check_if_exists(feature_class):
        logging.info('Raw tfidf features already created.')
        return

    for c in columns:

        tfidf_vectorizer = TfidfVectorizer(min_df=MIN_DF,
                                           max_df=MAX_DF,
                                           max_features=None,
                                           strip_accents='unicode',
                                           analyzer='word',
                                           token_pattern=TOKEN_PATTERN,
                                           ngram_range=NGRAM_RANGE,
                                           use_idf=1,
                                           smooth_idf=1,
                                           sublinear_tf=1,
                                           stop_words='english',
                                           norm=NORM,
                                           vocabulary=None)

        X = tfidf_vectorizer.fit_transform(df_all[c])
        logging.info('Shape of Tfidf transform matrix is %s' % str(X.shape))

        feature_class = 'raw_tfidf_%s' % c
        dump_features(feature_class, X)

    logging.info('Raw tfidf features are created and saved to pickle file.')


def create_common_vocabulary_raw_tfidf_features(df_all, col1, col2):

    logging.info('Creating common vocabulary raw tfidf features.')
    feature_class = 'common_vocabulary_raw_tfidf'
    if check_if_exists(feature_class):
        logging.info('Common vocabulary raw tfidf features already created.')
        return

    tfidf_vectorizer = TfidfVectorizer(min_df=MIN_DF,
                                       max_df=MAX_DF,
                                       max_features=None,
                                       strip_accents='unicode',
                                       analyzer='word',
                                       token_pattern=TOKEN_PATTERN,
                                       ngram_range=NGRAM_RANGE,
                                       use_idf=1,
                                       smooth_idf=1,
                                       sublinear_tf=1,
                                       stop_words='english',
                                       norm=NORM,
                                       vocabulary=None)
    documents = pd.concat([df_all[col1], df_all[col2]], axis=0)
    X = tfidf_vectorizer.fit_transform(documents)

    X_col1 = X[0:len(df_all)]
    X_col2 = X[len(df_all):2 * len(df_all)]

    res = X_col1.multiply(X_col2)

    dump_features(feature_class, res)
    logging.info('Common vocabulary Raw tfidf features are created and saved'
                 'to pickle file.')


def create_svd_tfidf_features(columns, n_components=N_COMPONENTS):

    logging.info('Creating svd tfidf features.')
    feature_class = 'svd_tfidf'
    if check_if_exists(feature_class):
        logging.info('SVD tfidf features already created.')
        return

    data = []
    svd = TruncatedSVD(n_components=n_components, n_iter=15)

    for c in columns:
        pickle_filename = 'raw_tfidf_%s_features.pickle' % c
        pickle_file = os.path.join(PICKLE_DIR, pickle_filename)

        with open(pickle_file, 'rb') as file:
            X = pickle.load(file)

        X_transformed = svd.fit_transform(X)
        svd_columns = ['tfidf_svd_' + c + '_' + str(i) for i in range(n_components)]
        data.append(pd.DataFrame(X_transformed, columns=svd_columns))

    df = pd.concat(data, axis=1)
    df.reset_index(inplace=True, drop=True)
    add_features(feature_class, df.columns.tolist())
    dump_features(feature_class, df)

    logging.info('Shape of svd tfidf features is %s' % str(df.shape))
    logging.info('Svd tfidf features are created.')


def create_common_vocabulary_svd_tfidf_features(n_components=2*N_COMPONENTS):

    logging.info('Creating common vocabulary svd tfidf features.')
    feature_class = 'common_vocabulary_svd_tfidf'
    if check_if_exists(feature_class):
        logging.info('Common Vocabulary SVD tfidf features already created.')
        return

    svd = TruncatedSVD(n_components=n_components, n_iter=15)

    pickle_filename = 'common_vocabulary_raw_tfidf_features.pickle'
    pickle_file = os.path.join(PICKLE_DIR, pickle_filename)

    with open(pickle_file, 'rb') as file:
        X = pickle.load(file)

    X_transformed = svd.fit_transform(X)
    svd_columns = ['common_vocabulary_tfidf_svd_' + str(i) for i in range(n_components)]
    data = pd.DataFrame(X_transformed, columns=svd_columns)

    add_features(feature_class, data.columns.tolist())
    dump_features(feature_class, data)

    logging.info('Shape of common vocabulary svd tfidf features is %s' % str(data.shape))
    logging.info('Common vocabulary SVD tfidf features are created.')
