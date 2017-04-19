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
from model_utils import tune_parameters, fit_and_predict, cross_validation
from model_utils import get_classifiers, get_param_grids

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])
PRED_DIR = os.path.join(OUTPUT_DIR, CONFIG['PRED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']

# Turn off/on parameters tuning and cross validation.
TUNE_PARAMETERS = False
DO_CROSS_VALIDATION = True


# TODO: this needs to be moved to utils type of file.
def save_predictions(preds, name):

    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)

    path = os.path.join(PRED_DIR, name + '.csv')
    preds.to_csv(path, index=False)


def generate_predictions(estimators, names, par_grids, class_features,
                         class_feature_names, test_ids, y_train, sparse=False):

    preds = test_ids.copy()

    # CV for parameter tuning. To speed up the process I am doing stratisfied
    # 2 iteration CV when tuning the parameters.
    cv1 = get_cv(y_train)

    # CV for cross validation. Must be KFold to create metafeatures.
    cv2 = get_cv(y_train, n_folds=5, type='kfold')

    for (class_feature, features_name) in zip(class_features, class_feature_names):

        logging.info('Reading data from files.')
        X_train, X_test = load_X(class_feature, len(y_train), sparse=sparse)
        logging.info('Shape of X_train: %s' % str(X_train.shape))
        logging.info('Data was succesfully read')

        for (estimator, par_grid, name) in zip(estimators, par_grids, names):
            filename = '_'.join((name, features_name))

            if TUNE_PARAMETERS:
                logging.info('Doing parameter tuning for %s model' % name)
                best_params, best_score = tune_parameters(estimator, name,
                                                          par_grid, X_train,
                                                          y_train, cv1)
                estimator.set_params(**best_params)
                logging.info('Finished parameter tuning for %s model' % name)

            if DO_CROSS_VALIDATION:
                logging.info('Doing cross validation for %s model' % name)
                cross_validation(estimator, X_train, y_train, cv2,
                                 filename=filename)
                logging.info('Finished cross validation for %s model' % name)

            logging.info('Fitting %s model' % name)
            preds['is_duplicate'] = (
                fit_and_predict(estimator, X_train, y_train, X_test))
            save_predictions(preds, filename)
            logging.info('Finished fitting %s model' % name)


def modelling():

    logging.info('MODELLING')

    test_ids = pd.read_csv(TEST_FILE, usecols=['test_id'], nrows=TEST_NROWS)
    y_train = pd.read_csv(TRAIN_FILE, usecols=['is_duplicate'],
                          nrows=TRAIN_NROWS)
    y_train = y_train['is_duplicate'].values

    # Generating predictions for logistic regression.
    names = ['LogisticRegression']
    estimators = get_classifiers(names)
    par_grids = get_param_grids(names)
    lr_class_features = [
        ['stemma_common_words', 'stemma_grouping', 'stemma_tfidf', 'stemma_distance_tfidf', 'stemma_word2vec',
         'stemma_raw_tfidf_question1', 'stemma_raw_tfidf_question2'],
        ['stemma_common_words', 'stemma_grouping', 'stemma_tfidf', 'stemma_distance_tfidf', 'stemma_word2vec',
         'stemma_common_vocabulary_raw_tfidf_question1',
         'stemma_common_vocabulary_raw_tfidf_question2'],
        ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
         'raw_tfidf_question1', 'raw_tfidf_question2'],
        ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
         'common_vocabulary_raw_tfidf_question1',
         'common_vocabulary_raw_tfidf_question2'],
        ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
         'raw_tfidf_question1', 'raw_tfidf_question2',
         'stemma_common_words', 'stemma_grouping', 'stemma_tfidf',
         'stemma_distance_tfidf', 'stemma_word2vec'
         ]
    ]
    class_feature_names = ['regular', 'common', 'stemma_regular',
                           'stemma_common', 'standard_stemma__mix']
    # generate_predictions(estimators, names, par_grids, lr_class_features,
    #                      class_feature_names, test_ids, y_train, sparse=True)

    # Generating predictions for XGBoost.
    names = ['XGBClassifier']
    estimators = get_classifiers(names)
    par_grids = get_param_grids(names)
    xgb_class_features = [
       # ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
       #  'svd_tfidf'],
       # ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
       #  'common_vocabulary_svd_tfidf'],
       # ['stemma_common_words', 'stemma_grouping', 'stemma_tfidf', 'stemma_distance_tfidf', 'stemma_word2vec',
      #  'stemma_svd_tfidf'],
      # ['stemma_common_words', 'stemma_grouping', 'stemma_tfidf', 'stemma_distance_tfidf', 'stemma_word2vec',
      #   'stemma_common_vocabulary_svd_tfidf'],
      ['stemma_stopwords_common_words', 'stemma_stopwords_grouping', 'stemma_stopwords_tfidf', 'stemma_stopwords_distance_tfidf', 'stemma_stopwords_word2vec',
       'stemma_stopwords_svd_tfidf'],
      ['common_words', 'grouping', 'tfidf', 'distance_tfidf', 'word2vec',
         'common_vocabulary_svd_tfidf',
         'stemma_common_words', 'stemma_grouping', 'stemma_tfidf',
         'stemma_distance_tfidf', 'stemma_word2vec',
      'stemma_stopwords_common_words', 'stemma_stopwords_grouping', 'stemma_stopwords_tfidf', 'stemma_stopwords_distance_tfidf', 'stemma_stopwords_word2vec'
        ]
    ]
    class_feature_names = ['stemma_stopwords_regular', 'standard_stemma_stopwords_mix']

    generate_predictions(estimators, names, par_grids, xgb_class_features,
                         class_feature_names, test_ids, y_train)

    logging.info('FINISHED MODELLING.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    modelling()
