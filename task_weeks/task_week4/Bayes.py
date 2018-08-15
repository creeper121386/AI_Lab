import numpy as np
import function as F 
normalize = False


class PredFunc(object):
    def __init__(self, miu, sigma, Pc, label):
        self.miu = miu
        self.sigma = sigma
        self.label = label
        self.Pc = Pc

    def cal(self, x):
        return self.Pc*np.prod(np.exp(-(x-self.miu)**2/(2*self.sigma**2))/(2.507*self.sigma))

def train(trainData, trainLabel):
    func = []
    num = len(trainLabel)
    newData, newLabel = F.divide(trainData, trainLabel)
    for C, y in zip(newData, newLabel):
        Pc = len(C)/num
        miu = np.average(C, axis=0)
        sigma = np.var(C, axis=0)
        func.append(PredFunc(miu, sigma, Pc, y))
    return func


trainData, trainLabel, _ = F.load(0, normalize)
testData, testLabel, _ = F.load(1, normalize)
func = train(trainData, trainLabel)
acc = F.test(testData, testLabel, func)
print(acc)
