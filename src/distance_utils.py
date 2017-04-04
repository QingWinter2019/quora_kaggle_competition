import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(v1, v2):
    return cosine_similarity(v1, v2)[0][0]


def rmse(v1, v2):
    diff = v2 - v1
    res = np.sqrt(np.mean(diff.multiply(diff)))
    return res
