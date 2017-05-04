import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

from globals import CONFIG
from feature_classification import add_features
from pickle_utils import check_if_exists, dump_features, load_features
from distance_utils import cosine_sim, rmse

# Global Tfidf parameters.
MAX_DF = 1.0
MIN_DF = 10
NGRAM_RANGE = (1, 2)
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
MAX_FEATURES = 200

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])


def create_most_common_words_features(df_all, col1, col2,
                                      max_features=MAX_FEATURES, pref=''):

    logging.info('Creating most common words features.')
    feature_class = pref + 'most_common_words'
    if check_if_exists(feature_class):
        logging.info('Most common words features already created.')
        return

    count_vectorizer = CountVectorizer(min_df=MIN_DF,
                                       max_df=MAX_DF,
                                       max_features=MAX_FEATURES,
                                       strip_accents='unicode',
                                       analyzer='word',
                                       token_pattern=TOKEN_PATTERN,
                                       ngram_range=NGRAM_RANGE,
                                       stop_words='english',
                                       binary=True,
                                       vocabulary=None)
    documents = pd.concat([df_all[col1], df_all[col2]], axis=0)
    X = count_vectorizer.fit_transform(documents)

    logging.debug(count_vectorizer.get_feature_names())

    X_col1 = X[0:len(df_all)]
    X_col2 = X[len(df_all):2 * len(df_all)]

    res = X_col1 + X_col2

    dump_features(feature_class, res)
    logging.info('Most common words features are created and saved to pickle file.')

