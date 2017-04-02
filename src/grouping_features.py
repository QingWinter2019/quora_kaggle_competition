import logging
import pandas as pd

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features


def create_grouping_features(df_all):

    logging.info('Creating grouping features.')
    feature_class = 'grouping_features'
    if check_if_exists(feature_class):
        logging.info('Grouping features already created.')
        return

    columns = ['distance1', 'distance2', 'absdistance1', 'absdistance2']
    df = load_features('common_words')[columns]
    df = pd.concat([df, df_all[['question1', 'question2']]], axis=1)
    df['q1count'] = df.groupby('question1')['question2'].transform('count')
    df['q2count'] = df.groupby('question2')['question1'].transform('count')
    df['q1count_gr_q2count'] = df['q1count'] > df['q2count']

    res = pd.DataFrame()

    for col in columns:
        df['min_group_q1_' + col] = df.groupby('question1')[col].transform(min)
        df['min_group_q2_' + col] = df.groupby('question1')[col].transform(min)
        df['min_' + col] = df.apply(
            lambda x: x['min_group_q1_' + col] if x['q1count_gr_q2count'] else
            x['min_group_q2_' + col], axis=1)
        df['max_group_q1_' + col] = df.groupby('question1')[col].transform(max)
        df['max_group_q2_' + col] = df.groupby('question1')[col].transform(max)
        df['max_' + col] = df.apply(
            lambda x: x['max_group_q1_' + col] if x['q1count_gr_q2count'] else
            x['max_group_q2_' + col], axis=1)

        res['min_' + col] = df['min_' + col]
        res['max_' + col] = df['max_' + col]
        res['rel_min_' + col] = df.apply(
            lambda x: x[col]/x['min_' + col] if x['min_' + col] != 0 else 0,
            axis=1)
        res['rel_max_' + col] = df.apply(
            lambda x: x[col] / x['max_' + col] if x['max_' + col] != 0 else 0,
            axis=1)

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Grouping features are created and saved to pickle file.')
