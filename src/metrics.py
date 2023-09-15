import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean


_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64


def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def compute_metrics(predict, target):
    return {
        'hellinger': np.mean([hellinger2(predict[i], target[i]) for i in range(predict.shape[0])]),
    }
