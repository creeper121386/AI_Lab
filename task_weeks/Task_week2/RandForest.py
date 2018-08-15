import DecisionTree as API
import matplotlib as plt
import numpy as np
treeNum = 30
dataNum = 1000  # 训练集样本个数
k = 3   # 选择k个特征比较信息熵


def load(f_name, divide, delimiter):
    data = np.loadtxt("/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/tree/" +
                      f_name, dtype=np.str, delimiter=delimiter)
    data = data[1:, :]
    train = data[:dataNum, :]
    test = data[:dataNum, :]
    if divide:
        trainLabel = train[:, -1:]
        trainData = train[:, :-1]
        testLabel = test[:, -1:]
        testData = test[:, :-1]
        return trainData, trainLabel, testData, testLabel
    else:
        return train, test


class Forest(object):
    def __init__(self, trainData, testData, trainLabel, testLabel):
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.testData = testData
        self.testLabel = testLabel
        self.trainIndex = []
        self.trees = []
        for _ in range(treeNum):
            self.trees.append([])
            self.trainIndex.append(set([np.random.randint(0, dataNum)
                                        for _ in range(dataNum)]))

    def train(self):
        for i in range(treeNum):
            feat = [0, 1, 2, 3, 4, 5]
            data = np.array([self.trainData[x] for x in self.trainIndex[i]])
            label = np.array([self.trainLabel[x] for x in self.trainIndex[i]])
            self.trees[i] = API.plant(data, label, feat)

    def test(self):
        count = 0
        num = len(self.testLabel)
        predLabel = np.array([API.pred(self.testData, T) for T in self.trees])
        finalPred = []
        for i in range(num):
            pred = [predLabel[j][i][1] for j in range(treeNum)]
            finalPred.append(max(pred ,key=pred.count))      
        for i in range(num):
            if self.testLabel[i] == finalPred[i]:
                count += 1
        acc = count/num
        return acc


trainData, trainLabel, testData, testLabel = load(
    'traindata.txt', divide=True, delimiter=' ')
forest = Forest(trainData, testData, trainLabel, testLabel)
forest.train()
print('[test over] \nacc =', forest.test())