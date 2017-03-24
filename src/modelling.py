"""
__file__

    modelling.py

__description__

    Generate predictions.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import os
import pandas as pd
from xgboost import XGBClassifier
import pickle
import re

from globals import CONFIG
from feature_classification import get_features


# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train_preprocess.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_preprocess.csv')
PRED_FILE = os.path.join(OUTPUT_DIR, 'pred.csv')


def modelling():

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    features = get_features()

    xgb_clf = XGBClassifier(base_score=0.5,
                            colsample_bylevel=1,
                            colsample_bytree=1,
                            gamma=0.7,
                            learning_rate=0.03,
                            max_delta_step=0,
                            max_depth=9,
                            min_child_weight=9.0,
                            missing=None,
                            n_estimators=430,
                            nthread=-1,
                            objective='binary:logistic',
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=2016,
                            silent=False,
                            subsample=0.9)
    xgb_clf.fit(train_data[features], train_data['is_duplicate'])

    test_data['is_duplicate'] = xgb_clf.predict_proba(test_data[features])[:, 1]

    test_data.rename(columns={'id': 'test_id'}, inplace=True)
    test_data[['test_id', 'is_duplicate']].to_csv(PRED_FILE, index=False)


if __name__ == '__main__':
    modelling()