import logging
import pandas as pd
import nltk

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features
import np_utils


def create_sentences(str_):

    sentences = nltk.sent_tokenize(str_)
    return sentences


def create_count_features(df_all, pref=''):

    logging.info('Creating count features.')
    feature_class = pref + 'count'
    if check_if_exists(feature_class):
        logging.info('Count features (%s) already created.' %
                     feature_class)
        return

    df_q1_q2 = df_all[['question1', 'question2']].reset_index(drop=True)
    df_q2_q1 = df_all[['question1', 'question2']].reset_index(drop=True)
    df_q2_q1.rename(columns={'question1': 'question2',
                             'question2': 'question1'})
    df = pd.concat([df_q1_q2, df_q2_q1], axis=0, ignore_index=True)

    # Create count of q1 and q2 features.
    res = pd.DataFrame()
    grouper1 = df.reset_index().groupby('question1')
    grouper2 = df.reset_index().groupby('question2')
    res['q1count'] = grouper1['question2'].transform('count')
    res['q2count'] = grouper2['question1'].transform('count')
    res['q1rank'] = grouper1['question2'].rank()
    res['q2rank'] = grouper2['question1'].rank()
    # res['hash1'] = grouper1['index'].transform(lambda x: x.iloc[0])
    # res['hash2'] = grouper2['index'].transform(lambda x: x.iloc[0])
    res = res[0:len(df_q1_q2)]

    # Number of sentences count.
    res['sent1count'] = df_q1_q2['question1'].apply(
        lambda x: len(create_sentences(x)))
    res['sent2count'] = df_q1_q2['question2'].apply(
        lambda x: len(create_sentences(x)))

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Count features are created and saved to pickle file.')
