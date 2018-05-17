import function as F
import numpy as np


class PredFunc(object):
    def __init__(self, ave, sigma, pi, label):
        self.ave = ave
        self.sigma = sigma
        self.pi = pi
        self.label = label

    def cal(self, x):
        return -0.5/self.sigma*np.dot((x-self.ave).T, x-self.ave)-0.5*np.log(np.abs(self.sigma))+np.log(self.pi)


def train(trainData, trainLabel):
    func = []
    num = len(trainLabel)
    newData, newLabel = F.divide(trainData, trainLabel)
    for C, y in zip(newData, newLabel):
        ave = np.average(C, axis=0)
        pi = len(C)/num
        sigma = np.var(C)
        func.append(PredFunc(ave, sigma, pi, y))
    return func


trainData, trainLabel, num = F.load(0, True)
testData, testLabel, _ = F.load(1, True)
func = train(trainData, trainLabel)
acc = F.test(testData, testLabel, func)
print(acc)
