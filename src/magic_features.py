import logging
import networkx as nx
import numpy as np
import pandas as pd

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features


def create_magic_features(df_all, pref=''):

    logging.info('Creating magic features.')
    feature_class = pref + 'magic'
    if check_if_exists(feature_class):
        logging.info('Magic features (%s) already created.' %
                     feature_class)
        return

    # 1. Creating questions dictionary: question -> hash_value.
    logging.debug('Creating questions dictionary...')
    questions1 = df_all[['question1', 'question2']].copy()
    questions2 = df_all[['question2', 'question1']].copy()
    questions2.rename(columns={'question1': 'question2', 'question2': 'question1'},
                      inplace=True)
    questions = questions1.append(questions2)
    questions.reset_index(inplace=True, drop=True)

    unique_questions = questions.drop_duplicates(subset=['question1'])
    unique_questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(unique_questions.index.values,
                               index=unique_questions['question1'].values).to_dict()
    # 2. Creating hash values.
    logging.debug('Creating hash dictionary...')
    # res = pd.DataFrame()
    questions['q1hash'] = questions['question1'].map(questions_dict)
    questions['q2hash'] = questions['question2'].map(questions_dict)

    # 3. Creating intersection features.
    logging.debug('Creating edges.')
    questions['l1hash'] = questions['q1hash'].apply(lambda x: [x])
    questions['l2hash'] = questions['q2hash'].apply(lambda x: [x])
    questions['edges1'] = questions.groupby('q1hash')['l2hash'].transform(sum)
    questions['edges2'] = questions.groupby('q2hash')['l1hash'].transform(sum)

    # 4
    wanted_cols = ['l1hash', 'l2hash', 'edges1', 'edges2', 'q1hash', 'q2hash']
    res = questions[wanted_cols].copy()[0:len(df_all)]

    # 3. Creating intersection features.
    logging.debug('Creating intersection features...')
    res['common_edges'] = res.apply(
        lambda x: len(set(x.edges1).intersection(set(x.edges2))), axis=1)

    # 4. Is question 2 ever appeared as question 1 column.
    logging.debug('Creating q2 in q1 feature...')
    questions1 = set(res['q1hash'].values)
    res['q2inq1'] = res['q2hash'].apply(lambda x: int(x in questions1))

    res.drop(['l1hash', 'l2hash', 'edges1', 'edges2'], axis=1, inplace=True)
    print(res.head())
    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Magic features are created and saved to pickle file.')
