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
    questions1 = df_all[['question1']].copy()
    questions2 = df_all[['question2']].copy()
    questions2.rename(columns={'question1': 'question2'}, inplace=True)
    questions = pd.concat([questions1, questions2], ignore_index=True)
    questions.drop_duplicates(inplace=True)
    questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(questions.index.values,
                               index=questions.values).to_dict()

    # 2. Creating hash values.
    logging.debug('Creating hash dictionary...')
    res = pd.DataFrame()
    res['q1hash'] = df_all['question1'].map(questions_dict)
    res['q2hash'] = df_all['question2'].map(questions_dict)

    # 3. Creating intersection features.
    logging.debug('Creating intersection features...')
    g = nx.Graph()
    g.add_nodes_from(res.q1hash)
    g.add_nodes_from(res.q2hash)
    edges = list(res[['q1hash', 'q2hash']].to_records(index=False))
    g.add_edges_from(edges)

    def inters_count(x, y):
        if x == y:
            return -1
        if (np.isnan(x) or np.isnan(y)):
            return -2
        neighbors_x = set(g.neighbors(x))
        neighbors_y = set(g.neighbors(y))
        return len(neighbors_x.intersection(neighbors_y))

    res['inters_count'] = res.apply(
        lambda row: inters_count(row.q1hash, row.q2hash), axis=1)

    # 4. Is question 2 ever appeared as question 1 column.
    logging.debug('Creating q2 in q1 feature...')
    questions1 = set(res['q1hash'].values)
    res['q2inq1'] = res['q2hash'].apply(lambda x: int(x in questions1))

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Magic features are created and saved to pickle file.')
