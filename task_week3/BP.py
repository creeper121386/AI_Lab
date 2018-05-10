import matplotlib as plt
import pandas as pd
import numpy as np
import math
batch = 4
epoch = 200
alpha = 0.01
size = [5, 10, 4]
normalize = False


class Net(object):
    def __init__(self, size):
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.random.randn(num) for num in size[1:]]
        '''self.weights = [np.ones((x, y))
                        for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.ones(num) for num in size[1:]]'''
        self.num = len(size)
        self.size = size

    def pred(self, feat):
        tmp = feat
        nerve = [feat]
        for i in range(self.num-1):
            tmp = sig(np.dot(tmp, self.weights[i])+self.bias[i])
            nerve.append(tmp)
        return nerve

    def update(self, label, nerve, alpha):
        sigma = (label-nerve[-1])
        deltaW = [x*0 for x in self.weights]
        deltaB = [x*0 for x in self.bias]
        for i in range(self.num-1):
            grad = nerve[-1-i]*(-nerve[-1-i]+1)*sigma
            gradMat = np.row_stack((grad,)*len(nerve[-2-i]))
            Bn = np.column_stack((nerve[-2-i],)*len(nerve[-1-i]))
            deltaW[-1-i] += alpha*gradMat*Bn
            deltaB[-1-i] += -alpha*grad
            sigma = np.sum(grad*self.weights[-1-i])
        return deltaW, deltaB

    def BP(self, trainData, trainLabel, epoch, batch, alpha):
        if not batch:
            batch = len(trainLabel)
        for j in range(epoch):
            for i in range(int(len(trainLabel)/batch)):
                deltaW = [x*0 for x in self.weights]
                deltaB = [x*0 for x in self.bias]
                batchData = trainData[i:i+batch, :]
                batchLabel = trainLabel[i:i+batch]
                nerve = [self.pred(x) for x in batchData]
                predLabel = [x[-1] for x in nerve]
                loss = np.sum((batchLabel-np.array(predLabel))**2)/batch
                for x, y in zip(batchLabel, nerve):
                    w, b = self.update(x, y, alpha)
                    for k in range(self.num-1):
                        deltaW[k] += w[k]
                        deltaB[k] += b[k]
                for k in range(self.num-1):
                    self.weights[k] += deltaW[k]/batch
                    self.bias[k] += deltaB[k]/batch
            print('training: epoch', j, 'loss:', loss)

    def AEBP(self, trainData, trainLabel, epoch, alpha):
        length = len(trainLabel)
        for j in range(epoch):
            deltaW = [x*0 for x in self.weights]
            deltaB = [x*0 for x in self.bias]
            nerve = [self.pred(x) for x in trainData]
            predLabel = [x[-1] for x in nerve]
            loss = np.sum((trainLabel-np.array(predLabel))**2)/length
            for x, y in zip(trainLabel, nerve):
                w, b = self.update(x, y, alpha)
                for k in range(self.num-1):
                    deltaW[k] += w[k]
                    deltaB[k] += b[k]
            for k in range(self.num-1):
                self.weights[k] += deltaW[k]/length
                self.bias[k] += deltaB[k]/length
            print('training: epoch', j, 'loss:', loss)

    def test(self, testData, testLabel):
        count = 0
        for x, y in zip(testData, testLabel):
            tmp = self.pred(x)[-1]
            Pred = np.argmax(tmp)
            if Pred == np.argmax(y):
                count += 1
        acc = count/len(testLabel)
        return acc


def load(sheetNo, normalize):
    df = pd.read_excel(
        "/media/why/DATA/why的程序测试/AI_Lab/Task/task_week3/data5.xls", sheetname=sheetNo)
    data = np.array(df.as_matrix())
    label = data[:, -1:]
    num = len(label)
    values = np.unique(label).tolist()
    label = [[1 if y == x else 0 for y in values] for x in label]
    data = np.array(data[:, :-1], dtype='float')
    # 特征归一化：
    if normalize:
        maxMat = np.row_stack((np.max(data, axis=0),)*num)
        minMat = np.row_stack((np.min(data, axis=0),)*num)
        data = (data-minMat)/(maxMat-minMat)
    return data, np.array(label)


def sig(x):
    return 1.0/(1.0+np.exp(-x))


def getArgs():
    while True:
        size = input(
            'enter the size of the net(except the last layer, splitted by space):')
        size = size.split()
        size = [int(x) for x in size]
        if not(size[0] == 5 and size[-1] == 4):
            print('Error: size mismatch.')
        else:
            break
    alpha = float(input('enter the learning rate:'))
    batch = int(input(
        'enter the size of mini-batch(if you don\'t want to use mini-batch, enter 0):'))
    epoch = int(input('enter the num of epoch:'))
    tmp = input('Do you want to do normalization?(y/n)')
    normalize = True if tmp == 'y' else False
    return size, batch, alpha, epoch, normalize


# size, batch, alpha, epoch, normalize = getArgs()
trainData, trainLabel = load(1, normalize)
testData, testLabel = load(2, normalize)
nn = Net(size)
nn.AEBP(trainData, trainLabel, epoch, alpha)
# nn.BP(trainData, trainLabel, epoch, batch, alpha)
# acc = nn.test(trainData, trainLabel)
acc = nn.test(testData, testLabel)
print('test finished. acc:', acc*100, '%')
