import logging
import numpy as np
import os
import pandas as pd
import pickle

from globals import CONFIG
from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features
from model_utils import rescale_preds, score, A, B
from stacking import filenames_in_dir

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PRED_DIR = os.path.join(OUTPUT_DIR, CONFIG['PRED_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])
METAFEATURES_DIR = os.path.join(PICKLE_DIR, CONFIG['METAFEATURES_DIR'])


def create_logistic_features(metafeatures_dir=METAFEATURES_DIR,
                             preds_dir=PRED_DIR, pref=''):

    logging.info('Creating logistic features.')
    feature_class = pref + 'logistic'
    if check_if_exists(feature_class):
        logging.info('Logistic features (%s) already created.' %
                     feature_class)
        return

    metafeatures_filenames = filenames_in_dir(metafeatures_dir, '.pickle')
    preds_filenames = filenames_in_dir(preds_dir, '.csv')
    common_filenames = set(metafeatures_filenames).intersection(set(preds_filenames))
    common_filenames = sorted(common_filenames)

    # We are only interested in logistic metafeatures.
    common_filenames = [f for f in common_filenames if f.startswith('Logistic')]

    train_data = []
    for filename in common_filenames:
        # Only logistic regression use as features.
        if not filename.startswith('Logistic'):
            continue

        with open((os.path.join(metafeatures_dir, filename + '.pickle')), 'rb') as file:
            try:
                metafeature = np.sum(pickle.load(file), axis=1)
            except:
                metafeature = pickle.load(file)
            metafeature = rescale_preds(metafeature, a=B, b=A)
            train_data.append(metafeature)

    train_data = np.stack(train_data, axis=1)
    train_data = pd.DataFrame(train_data, columns=common_filenames)

    # Load preds.
    test_data = []
    for filename in common_filenames:
        file = os.path.join(preds_dir, filename + '.csv')
        preds = pd.read_csv(file, usecols=['is_duplicate'])
        # We need to rescale predictions back ot avoid double rescaling.
        # TODO: think about a better way to do it.
        preds = rescale_preds(preds, a=B, b=A)
        test_data.append(preds.values)

    test_data = np.concatenate(test_data, axis=1)
    test_data = pd.DataFrame(test_data, columns=common_filenames)

    data = pd.concat([train_data, test_data])

    add_features(feature_class, common_filenames)
    dump_features(feature_class, data)
    logging.info('Logistic features are created and saved to pickle file.')
