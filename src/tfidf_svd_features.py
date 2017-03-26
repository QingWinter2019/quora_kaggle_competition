import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import pickle

from globals import CONFIG
from feature_classification import add_features

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

# Global files.
TFIDF_TRANSFORM_X_FILE = os.path.join(PICKLE_DIR, 'tfidf_X.pickle')


def add_tfidf_svd_features(df_all, columns):

    logging.info('Creating tfidf svd features, it may take some time')

    df = pd.DataFrame()
    df['id'] = df_all['id']

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
        #logging.info('Vocabulary')
        #logging.info(tfidf_vectorizer.vocabulary_)
        with open(TFIDF_TRANSFORM_X_FILE, 'wb') as file:
            pickle.dump(X, file, pickle.HIGHEST_PROTOCOL)

        svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=15)
        X = svd.fit_transform(X)

        svd_columns = ['tfidf_svd_' + c + str(i) for i in range(N_COMPONENTS)]
        df[svd_columns] = pd.DataFrame(X, columns=svd_columns)

    add_features('tfidf_svd', df.columns.tolist())
    logging.info('Tfidf svd features are created.')

    df.drop('id', axis=1, inplace=True)
    df_all = pd.concat([df_all, df], axis=1)

    return df_all
