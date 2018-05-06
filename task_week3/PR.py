import matplotlib as plt
import pandas as pd
import numpy as np
import math
batch = 8
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
        nerve = [input]
        for i in range(self.num-1):
            tmp = sig(np.dot(tmp, self.weights[i])+self.bias[i])
            nerve.append(tmp)
        return nerve

    def update(self, label, nerve, alpha):
        sigma = (label-nerve[-1])
        for i in range(self.num-1):
            grad = nerve[-1-i]*(-nerve[-1-i]+1)*sigma
            gradMat = np.row_stack((grad,)*len(nerve[-2-i]))
            Bn = np.column_stack((nerve[-2-i],)*len(nerve[-1-i]))
            self.weights[-1-i] += alpha*gradMat*Bn
            self.bias[-1-i] += -alpha*grad
            sigma = np.sum(grad*self.weights[-1-i])

    def BP(self, trainData, trainLabel, epoch, batch, alpha):
        for j in range(epoch):
            if batch:
                for i in range(int(len(trainLabel)/batch)):
                    batchData = trainData[i:i+batch, :]
                    batchLabel = trainLabel[i:i+batch]
                    nerve = [self.pred(x) for x in batchData]
                    predLabel = [x[-1] for x in nerve]
                    loss = np.sum((batchLabel-np.array(predLabel))**2)/2
                    for x, y in zip(batchLabel, nerve):
                        self.update(x, y, alpha)
                    print('training: epoch', epoch, 'loss:', loss)
                else:
                    nerve = [self.pred(x) for x in trainData]
                    for x, y in zip(trainLabel, nerve):
                        self.update(x, y, alpha)
                    print('train finished,loss:', loss)

    def test():
        pass

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


testData, teatLabel, cateNum = load(2)
trainData, trainLabel, cateNum = load(1)
nn = Net((5, 10, cateNum))
nn.BP(trainData, trainLabel, epoch, batch, alpha)
