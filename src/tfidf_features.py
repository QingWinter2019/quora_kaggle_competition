import logging
import pandas as pd

from pickle_utils import dump_features, check_if_exists
from feature_classification import add_features
from tfidf import create_idf, tfidf1, tfidf2

TFIDF_ANALYSIS = True
TFIDF2 = True


def create_tfidf_features(df_all, columns, qcol, unique=False):

    logging.info('Creating tfidf features')
    feature_class = 'tfidf'
    if check_if_exists(feature_class):
        logging.info('Tfidf features already created.')
        return

    df = pd.DataFrame()
    df['id'] = df_all['id']

    if TFIDF_ANALYSIS:

        for c in columns:

            logging.info("Doing TFIDF Analysis, it may take some time")
            if unique:
                create_idf(df_all[c].unique())
            else:
                create_idf(df_all[c])

            # different types of tfidf
            types = ('binary', 'freq', 'log_freq', 'dnorm')

            # different ways of aggregating term tfidf in query
            indexes, prefixes = (0, 1, 2), ('s', 'm', 'a')

            # two different functions - one exact match, other - common words
            funcs, suffixes = [tfidf1, tfidf2], ('1', '2')

            for (func, suffix) in zip(funcs, suffixes):
                if (func == tfidf2) and (not TFIDF2):
                    continue

                df['temp'] = df_all.apply(lambda x: func(str(x[qcol]), str(x[c]), type='all'), axis=1)

                ind = 0
                for t in types:
                    for prefix in prefixes:
                        name = qcol + prefix + t + '_tfidf_' + c + '_' + suffix
                        df[name] = df['temp'].map(lambda x: x[ind])
                        ind += 1

            df.drop(['temp'], axis=1, inplace=True)

            logging.info('TFIDF analysis is finished')

    df.drop('id', axis=1, inplace=True)
    add_features(feature_class, df.columns.tolist())
    dump_features(feature_class, df)
    logging.info('Tfidf features are created and saved to pickle file.')
