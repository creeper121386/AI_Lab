import matplotlib as plt
import pandas as pd
import numpy as np
import math
size = (5, 10, 1)
batchSize = 4
alpha = 0.01
epoch = 100


class Net(object):
    def __init__(self, size):
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.random.random(num) for num in size[1:]]
        self.num = len(size)
        self.size = size

    def pred(self, input):
        tmp = input
        for i in range(self.num):
            tmp = sig(np.dot(np.transpose(self.weights[i]), tmp)+np.transpose(self.bias[i]))
            '''weight = np.transpose(self.weights[i])
            data = tmp
            b = np.transpose(self.bias[i])
            tmp = sig(np.dot(self.weights[i], np.transpose(tmp))+self.bias[i])'''
        return tmp

    def GD(self, trainData, trainLabel, epoch, batchSize, alpha):
        for i in range(len(trainLabel)/batchSize):
            batchData = trainData[:, i:i+batchSize]
            batchLabel = trainLabel[:, i:i+batchSize]
            predLabel = [self.pred(x) for x in batchData]
            loss = np.fabs(batchLabel-np.array(predLabel))





def load(sheetNo):
    df = pd.read_excel(
        "/media/why/DATA/why的程序测试/AI_Lab/Task/task_week3/data5.xls", sheetname=sheetNo)
    data = np.array(df.as_matrix())
    label = data[:, -1:]
    values = np.unique(label).tolist()
    label = [values.index(x) for x in label]
    return np.array(data[:, :-1],dtype='float'), label


def sig(x):
    return 1.0/(1.0+np.exp(-x))


trainData, trainLabel = load(1)
nn = Net(size)
tmp=nn.pred(trainData[1])
pass
