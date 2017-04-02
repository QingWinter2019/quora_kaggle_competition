import logging
import os
import pandas as pd
import pickle

from globals import CONFIG

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])


def load_X(feature_classes, train_size):

    logging.info('Loading features %s' % feature_classes)
    data = [load_features(feature_class) for feature_class in feature_classes]

    for df in data:
        df.reset_index(inplace=True, drop=True)

    df = pd.concat(data, axis=1, ignore_index=True)
    logging.info('Features are loaded.')
    return df[:train_size], df[train_size:]


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
