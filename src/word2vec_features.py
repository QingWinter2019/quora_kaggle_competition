import gensim
import logging
import numpy as np
import os
import pandas as pd

from pickle_utils import check_if_exists, dump_features, load_features
from feature_classification import add_features

import np_utils
import distance_utils

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
MODELS_DIR = os.path.join(BASE_DIR, 'data/pretrained_models')


class Word2VecEstimator:

    def __init__(self, word2vec_model):
        self.model = word2vec_model.model
        self.vector_size = 300  # TODO: fix this to be derived from model

    def _get_valid_word_list(self, words):
        return [w for w in words if w in self.model]

    def _get_importance(self, text1, text2):
        len_prev_1 = len(text1)
        len_prev_2 = len(text2)
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = np_utils.try_to_divide(len1+len2, len_prev_1+len_prev_2)
        return imp

    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def get_n_similarity(self, text1, text2):
        lst1 = self._get_valid_word_list(text1)
        lst2 = self._get_valid_word_list(text2)
        if len(lst1) > 0 and len(lst2) > 0:
            return self.model.n_similarity(lst1, lst2)
        else:
            return 0

    def get_n_similarity_imp(self, text1, text2):
        sim = self.get_n_similarity(text1, text2)
        imp = self._get_importance(text1, text2)
        return sim * imp

    def get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return distance_utils.rmse(centroid1, centroid2)

    def get_centroid_rmse_imp(self, text1, text2):
        rmse = self.get_centroid_rmse(text1, text2)
        imp = self._get_importance(text1, text2)
        return rmse * imp


class Word2VecModel:

    def __init__(self, name, path=None, corpus=None):

        self.name = name

        logging.info(path)
        # Load pre-trained model if path is not None.
        if path is not None:
            try:
                if '.bin' in path:
                    self.model = gensim.models.KeyedVectors.load_word2vec_format(
                        path, binary=True)
                elif '.txt' in path:
                    self.model = gensim.models.Word2Vec.load_word2vec_format(
                        path, binary=False)
                else:
                    self.model = gensim.models.Word2Vec.load(path)
            except:
                raise ValueError('Unknown file format for pre-trained model')

        # Build your own model.
        if corpus is not None:
            self.model = gensim.models.Word2Vec(corpus, sg=1, window=10,
                                                sample=1e-5, negative=5,
                                                size=300)

    def n_similarity(self, text1, text2):
        return self.model.n_similarity(text1, text2)

    def __getitem__(self, key):
        return self.model[key]


def create_word2vec_features(data, col1, col2, pref=''):

    logging.info('Creating Word2Vec features.')
    feature_class = pref + 'word2vec'
    if check_if_exists(feature_class):
        logging.info('Word2Vec features are already created.')
        return

    models = []

    # Create our own model.
    corpus = list(data[col1]) + list(data[col2])
    models.append(Word2VecModel(corpus=corpus, name='Corpus'))

    # Load pre-trained models.
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.txt') or file.endswith('.bin'):
            models.append(Word2VecModel(path=os.path.join(MODELS_DIR, file),
                                        name=file.split('.', 1)[0]))

    res = pd.DataFrame()
    for model in models:
        estimator = Word2VecEstimator(model)
        res['%s_n_similarity' % model.name] = data.apply(
            lambda x: estimator.get_n_similarity(x[col1], x[col2]), axis=1)
        res['%s_n_similarity_imp' % model.name] = data.apply(
            lambda x: estimator.get_n_similarity_imp(x[col1], x[col2]), axis=1)
        res['%s_centroid_rmse' % model.name] = data.apply(
            lambda x: estimator.get_centroid_rmse(x[col1], x[col2]), axis=1)
        res['%s_centroid_rmse_imp' % model.name] = data.apply(
            lambda x: estimator.get_centroid_rmse_imp(x[col1], x[col2]), axis=1)

    add_features(feature_class, res.columns.tolist())
    dump_features(feature_class, res)
    logging.info('Word2Vec features are created and saved to pickle file.')
