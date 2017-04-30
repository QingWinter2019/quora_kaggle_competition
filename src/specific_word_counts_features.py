import logging
import pandas as pd

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features

SPECIFIC_WORDS = ['how', 'why', 'where', 'who', 'what', 'which']
NEGATION_WORDS = ["n't", 'not']


def create_specific_word_counts(df_all, specific_words=SPECIFIC_WORDS, pref=''):

    logging.info('Creating specific word features.')
    feature_class = pref + 'specific_words'
    if check_if_exists(feature_class):
        logging.info('Specific word features (%s) already created.' %
                     feature_class)
        return

    # Doing some preprocessing to not relly of whether data is supplied
    # preprocessed already.
    df_all['question1'] = df_all['question1'].apply(lambda x: str(x).lower())
    df_all['question2'] = df_all['question2'].apply(lambda x: str(x).lower())

    res = pd.DataFrame()
    for word in specific_words:
        res[word + 'in_q1'] = df_all['question1'].apply(lambda x: int(word in x))
        res[word + 'in_q2'] = df_all['question2'].apply(lambda x: int(word in x))
        res[word] = res[word + 'in_q1'] + res[word + 'in_q2']

    res['neg_in_q1'], res['neg_in_q2'], res['neg'] = 0, 0, 0
    for word in NEGATION_WORDS:
        res['neg_in_q1'] = res['neg_in_q1'] + df_all['question1'].apply(
            lambda x: int(word in x))
        res['neg_in_q2'] = res['neg_in_q2'] + df_all['question2'].apply(
            lambda x: int(word in x))
    res['neg'] = res['neg_in_q1'] + res['neg_in_q2']

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Specific word counts features are created and saved to pickle file.')
