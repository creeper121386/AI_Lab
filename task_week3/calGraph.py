import numpy as np


class calGraph(object):
    def __init__(self, data):
        self.data = data


class node(object):
    Next = None
    pre = None
    forwdValue = None
    backValue = None


class mutiply(node):
    id = 0

    def __init__(self, x, y):
        self.forwdValue = x*y
        self.backValue = (y, x)


class plus(node):
    id = 1

    def __init__(self, x, y):
        self.forwdValue = x+y
        self.backValue = (1, 1)


class exp(node):
    id = 2

    def __init__(self, x):
        self.forwdValue = np.exp(x)
        self.backValue = np.exp(x)


class log(node):
    id = 3

    def __init__(self, x):
        self.forwdValue = np.log(x)
        self.backValue = 1/x


class power(node):
    id = 4

    def __init__(self, x, index):
        self.forwdValue = x**index
        self.backValue = index*x**index


class sigmoid(node):
    id = 5

    def __init__(self, x):
        self.forwdValue = 1/(1+np.exp(-x))
        self.backValue = 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))


class reLU(node):
    id = 6

    def __init__(self, x):
        self.forwdValue = np.maximum(x, 0)
        self.backValue = 1 if x > 0 else 0


class dot(node):

    def __init__(self, x, y):
        self.forwdValue = np.dot(x, y)
        self.backValue = np.exp(x)


graph = calGraph(0)
