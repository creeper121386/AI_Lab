import matplotlib as plt
import pandas as pd
import numpy as np
import math
batchSize = 8
alpha = 0.01
epoch = 100


class Net(object):
    def __init__(self, size):
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.random.randn(num) for num in size[1:]]
        self.num = len(size)
        self.size = size

    def pred(self, input):
        tmp = input
        for i in range(self.num-1):
            tmp = sig(np.dot(tmp, self.weights[i])+self.bias[i])
        return tmp

    def BP(self, trainData, trainLabel, epoch, batchSize, alpha):
        for j in range(epoch):
            for i in range(int(len(trainLabel)/batchSize)):
                batchData = trainData[i:i+batchSize, :]
                batchLabel = trainLabel[i:i+batchSize]
                predLabel = [self.pred(x) for x in batchData]
                loss = np.sum((batchLabel-np.array(predLabel))**2)/2



                self.weights+=deltaW
                self.bias+=deltaB

def load(sheetNo):
    df = pd.read_excel(
        "/media/why/DATA/why的程序测试/AI_Lab/Task/task_week3/data5.xls", sheetname=sheetNo)
    data = np.array(df.as_matrix())
    label = data[:, -1:]
    values = np.unique(label).tolist()
    label = [[1 if y == x else 0 for y in values] for x in label]
    return np.array(data[:, :-1], dtype='float'), np.array(label), len(values)


def sig(x):
    return 1.0/(1.0+np.exp(-x))


trainData, trainLabel, cateNum = load(1)
nn = Net((5, 10, cateNum))
nn.pred(trainData[1])
nn.BP(trainData, trainLabel, epoch, batchSize, alpha)
