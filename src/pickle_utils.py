import logging
import os
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from globals import CONFIG

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])


def load_X(feature_classes, train_size, sparse=False, norm=True):

    logging.info('Loading features %s' % feature_classes)
    data = [load_features(feature_class) for feature_class in feature_classes]

    if sparse:
        data = [sp.csr_matrix(features) for features in data]
        res = sp.hstack(data)
        if norm:
            res = normalize(res, norm='l1', axis=1)
    else:
        for df in data:
            df.reset_index(inplace=True, drop=True)
        res = pd.concat(data, axis=1, ignore_index=True).values

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


def check_if_exists(feature_class):

    pickle_file = os.path.join(PICKLE_DIR, '%s_features.pickle' % feature_class)
    return bool(os.path.exists(pickle_file))
