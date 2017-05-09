import logging
import pandas as pd

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features
import np_utils


def create_grouping_features(df_all, pref=''):

    logging.info('Creating grouping features.')
    feature_class = pref + 'grouping'
    if check_if_exists(feature_class):
        logging.info('Grouping features (%s) already created.' %
                     feature_class)
        return

    columns = ['distance1', 'distance2', 'absdistance1', 'absdistance2']
    common_words = (
        load_features(pref + 'common_words')[columns].reset_index(drop=True))

    if check_if_exists(pref + 'distance_tfidf'):
        distance_tfidf_features = (
            load_features(pref + 'distance_tfidf').reset_index(drop=True))
        columns += distance_tfidf_features.columns.tolist()
    else:
        distance_tfidf_features = pd.DataFrame()

    if check_if_exists(pref + 'word2vec'):
        word2vec_features = load_features(pref + 'word2vec').reset_index(drop=True)
        columns += word2vec_features.columns.tolist()
    else:
        word2vec_features = pd.DataFrame()

    if check_if_exists(pref + 'wordnet'):
        wordnet_features = load_features(pref + 'wordnet').reset_index(drop=True)
        columns += wordnet_features.columns.tolist()
    else:
        wordnet_features = pd.DataFrame()

    df_q1_q2 = pd.concat([common_words,
                          distance_tfidf_features,
                          word2vec_features,
                          wordnet_features,
                          df_all[['question1', 'question2']].reset_index(drop=True)],
                         axis=1)
    df_q2_q1 = pd.concat([common_words,
                          distance_tfidf_features,
                          word2vec_features,
                          wordnet_features,
                          df_all[['question2', 'question1']].reset_index(drop=True)],
                         axis=1)
    df_q2_q1.rename(columns={'question1': 'question2',
                             'question2': 'question1'})
    df = pd.concat([df_q1_q2, df_q2_q1], axis=0, ignore_index=True)

    # GroupBy objects.
    groupby_q1 = df.groupby('question1')
    groupby_q2 = df.groupby('question2')

    df['q1count'] = groupby_q1['question2'].transform('count')
    df['q2count'] = groupby_q2['question1'].transform('count')
    inds_q1_gr_q2 = (df['q1count'] > df['q2count'])[0:len(df_q1_q2)]
    inds_q2_gr_q1 = ~inds_q1_gr_q2

    res = pd.DataFrame()

    groupers = ['min', 'max', 'mean']
    for grouper in groupers:
        for col in columns:

            res[grouper + '_by_q1_' + col] = (
                groupby_q1[col].transform(grouper)[0:len(df_q1_q2)])
            res[grouper + '_by_q2_' + col] = (
                groupby_q2[col].transform(grouper)[0:len(df_q1_q2)])

            res[col] = df[col][0:len(df_q1_q2)]
            res['rel_q1_' + col] = res.apply(
                lambda x: np_utils.try_to_divide(x[col],
                                                 x[grouper + '_by_q1_' + col]),
                axis=1)
            res['req_q2_' + col] = res.apply(
                lambda x: np_utils.try_to_divide(x[col],
                                                 x[grouper + '_by_q2_' + col]),
                axis=1)

            res[grouper + '_by_' + col] = 0
            res[grouper + '_by_' + col][inds_q1_gr_q2] = res[
                grouper + '_by_q1_' + col]
            res[grouper + '_by_' + col][inds_q2_gr_q1] = res[
                grouper + '_by_q2_' + col]
            res['rel_' + col] = res.apply(
                lambda x: np_utils.try_to_divide(x[col],
                                                 x[grouper + '_by_' + col]),
                axis=1)

            del res[col]

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Grouping features are created and saved to pickle file.')
