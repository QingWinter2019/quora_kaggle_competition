"""
__file__

    cv_utils.py

__description__

    This file contains cross validation utils.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >
"""

from sklearn.cross_validation import StratifiedKFold, KFold
import random

from globals import CONFIG

# Global CV.
CV = []


def get_cv(y):
    '''
    :param
    :return:
    '''

    global CV

    if not CV:

        random.seed(CONFIG['RANDOM_SEED'])
        rs = [random.randrange(0, CONFIG['MAX_RANDOM_SEED']) for i in range(CONFIG['REPETITION_NUM'])]

        if CONFIG['CV_TYPE'] == 'KFOLD':
            CV = myKFold(len(y), CONFIG['FOLDS_NUM'], rseeds=rs)
        else:  # STRATIFIED
            CV = myStratifiedKFold(y, CONFIG['FOLDS_NUM'], rseeds=rs)

    return CV


def myStratifiedKFold(y, n_folds, shuffle=True, rseeds=[1]):
    """
    :param y:
    :param n_folds:
    :param n_rep:
    :param rs:
    :return:
    """

    cv = []
    for rs in rseeds:
        stratified_cv = StratifiedKFold(y,
                                        n_folds=n_folds,
                                        shuffle=shuffle,
                                        random_state=rs)
        for train_ind, test_ind in stratified_cv:
            cv.append([train_ind, test_ind])

    return cv


def myKFold(n, n_folds, shuffle=True, rseeds=[1]):
    """
    :param n: Total number of elements
    :param n_folds: Number of folds
    :param rs: When shuffle=True, pseudo-random generator state used for shuffling
    :return:
    """

    cv = []
    for rs in rseeds:
        kfold_cv = KFold(n=n, n_folds=n_folds, shuffle=shuffle, random_state=rs)
        for train_ind, test_ind in kfold_cv:
            cv.append([train_ind, test_ind])

    return cv
