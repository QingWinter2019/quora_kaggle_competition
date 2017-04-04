"""
__file__

    modelling.py

__description__

    Generate predictions.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import logging
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

from globals import CONFIG
from feature_classification import get_features, get_feature_classes
from pickle_utils import load_X
from cv_utils import get_cv

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
PRED_FILE = os.path.join(OUTPUT_DIR, 'pred.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']


def modelling():

    logging.info('MODELLING')

    class_features = [
        # ['raw_tfidf_question1', 'raw_tfidf_question2', 'common_words', 'grouping_features', 'tfidf']]
        ['common_words', 'grouping', 'tfidf', 'svd_tfidf', 'distance_tfidf']]

    pred_files = [
        # os.path.join(OUTPUT_DIR, 'pred_common.csv'),
        os.path.join(OUTPUT_DIR, 'preds.csv')
    ]

    preds = pd.read_csv(TEST_FILE,
                        usecols=['test_id'],
                        nrows=TEST_NROWS)
    y_train = pd.read_csv(TRAIN_FILE,
                          usecols=['is_duplicate'],
                          nrows=TRAIN_NROWS)
    y_train = y_train['is_duplicate'].values

    for (class_feature, pred_file) in zip(class_features, pred_files):

        logging.info('Reading data from files.')
        X_train, X_test = load_X(class_feature, len(y_train), sparse=False)
        logging.info('Data was succesfully read')

        clf = XGBClassifier(base_score=0.5,
                            colsample_bylevel=1,
                            colsample_bytree=0.9,
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
                            silent=True,
                            subsample=0.9)

        # uncomment for logistic regression
        # lr = LogisticRegression(C=10)

        cv = get_cv(y_train)
        logging.info('Shape of X_train: %s' % str(X_train.shape))
        logging.info('Doing Cross Validation')
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 scoring='log_loss',
                                 cv=cv,
                                 n_jobs=1)
        logging.info(scores)
        logging.info('Finished Cross Validation')

        logging.info('Fitting the model')
        clf.fit(X_train, y_train)
        logging.info('Finished fitting the model')

        logging.info('Generating model predictions')
        preds['is_duplicate'] = clf.predict_proba(X_test)[:, 1]
        preds.to_csv(pred_file, index=False)

    logging.info('FINISHED MODELLING.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    modelling()
