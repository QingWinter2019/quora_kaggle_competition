import logging
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import normalize, StandardScaler

from globals import CONFIG

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])
METAFEATURES_DIR = os.path.join(PICKLE_DIR, CONFIG['METAFEATURES_DIR'])
FEATURE_IMPORTANCES_DIR = os.path.join(PICKLE_DIR, CONFIG['FEATURE_IMPORTANCES_DIR'])


def load_X(feature_classes, train_size, sparse=False, norm=True):

    logging.info('Loading features %s' % feature_classes)
    data = [load_features(feature_class) for feature_class in feature_classes]

    if sparse:
        data = [sp.csr_matrix(features) for features in data]
        res = sp.hstack(data, format='csr')
        if norm:
            scaler = StandardScaler(with_mean=False)
            res = scaler.fit_transform(res)
    else:
        for df in data:
            try:
                df.reset_index(inplace=True, drop=True)
            except:
                df = pd.DataFrame(df)
        res = np.concatenate([df.values for df in data], axis=1)

        cols = []
        for (fc, df) in zip(feature_classes, data):
            new_cols = [fc + '$' + c for c in df.columns]
            cols += new_cols
        res = pd.DataFrame(res, columns=cols)

    logging.info('Features are loaded.')
    return res[:train_size], res[train_size:]


def load_features(feature_class):

    pickle_file = os.path.join(PICKLE_DIR, '%s_features.pickle' % feature_class)
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    return data


def dump_features(feature_class, data):

    pickle_file = os.path.join(PICKLE_DIR, '%s_features.pickle' % feature_class)
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def dump_feature_importances(feature_importances, filename):

    if not os.path.exists(FEATURE_IMPORTANCES_DIR):
        os.makedirs(FEATURE_IMPORTANCES_DIR)

    with open(os.path.join(FEATURE_IMPORTANCES_DIR, filename + '.pickle'), 'wb') as file:
        pickle.dump(feature_importances, file, pickle.HIGHEST_PROTOCOL)


def dump_metafeatures(metafeatures, filename):

    if not os.path.exists(METAFEATURES_DIR):
        os.makedirs(METAFEATURES_DIR)

    with open(os.path.join(METAFEATURES_DIR, filename + '.pickle'), 'wb') as file:
        pickle.dump(metafeatures, file, pickle.HIGHEST_PROTOCOL)


def check_if_exists(feature_class):

    pickle_file = os.path.join(PICKLE_DIR, '%s_features.pickle' % feature_class)
    return bool(os.path.exists(pickle_file))
