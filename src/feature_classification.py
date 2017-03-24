"""
__file__

    feature_classification.py

__description__

    This file provides utils for feature classification.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import os
import pickle
from globals import CONFIG


BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_FILENAME = 'feature_classes.pickled'
PICKLE_FILE = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'], PICKLE_FILENAME)

# List of feature classes.
FEATURE_CLASSES = []
# Dictionary feature_class: list of features.
FEATURE_DICT = dict()


def _load_data():
    """Load features (classes and dict) from the pickle file to global variables.
    """
    global FEATURE_CLASSES, FEATURE_DICT
    # Check whether there is a need to read a file.
    if (len(FEATURE_CLASSES) == 0) and os.path.isfile(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as file:
            data = pickle.load(file)
            FEATURE_CLASSES = data['feature_classes']
            FEATURE_DICT = data['feature_dict']

    return FEATURE_CLASSES


def dump_features():
    """Save features (classes and dict) to the pickle file."""

    with open(PICKLE_FILE, 'wb') as file:
        data = {'feature_classes': FEATURE_CLASSES,
                'feature_dict': FEATURE_DICT}
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def get_feature_classes():
    """Return feature classes."""

    _load_data()
    return FEATURE_CLASSES


def get_features(include='all', exclude='none'):
    """
    :param include: list or 'all'
                    If 'all' then all features are include in the analysis
                    Otherwise only features that from classes in the list are
                    included
    :param exclude: list of 'none'
                    If 'none' then neither features are excluded
                    Otherwise features from classes in the are not included
    :return: list of features sorted by alphabetic order
    """

    _load_data()

    assert ((include == 'all') or set(include).issubset(set(FEATURE_CLASSES)))
    assert ((exclude == 'none') or set(exclude).issubset(set(FEATURE_CLASSES)))

    if include == 'all':
        include = FEATURE_CLASSES

    if exclude == 'none':
        exclude = []

    # Create set of features.
    features = set()
    # Add features from include.
    for feature_class in include:
        features = features.union(set(FEATURE_DICT[feature_class]))
    # Remove features from exclude.
    for feature_class in exclude:
        features = features.difference(set(FEATURE_DICT[feature_class]))

    # Sort features for reproducibility of results.
    features = list(features)
    features.sort()

    return features


def add_features(f_class, features):
    """ Add f_class and features to classes and dict.
    :param f_class: the name of feature class
    :param features: list of features
    :return:
    """

    assert_message = f_class + ' already exists in the list'
    assert f_class not in FEATURE_CLASSES, assert_message

    FEATURE_CLASSES.append(f_class)
    FEATURE_DICT[f_class] = features
