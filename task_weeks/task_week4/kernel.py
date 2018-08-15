import numpy as np
theta = -1  # sigmoid核参数
sigma = 2
beta = 1    # sigmoid核参数
d = 3   # 多项式核的指数


def liner(x, y):
    return np.dot(x, y)


def multi(x, y):
    return (np.dot(x, y))**d


def Gauss(x, y):
    return np.exp(-np.linalg.norm(x-y.transpose())**2/(2*sigma**2))


def Laplace(x, y):
    return np.exp(-(np.linalg.norm(x-y.transpose()))/sigma)


def sigmoid(x, y):
    return np.tanh(beta*np.dot(x, y)+theta)
