import numpy as np
import pandas as pd
C = 1e5
loopTimes = 10


def load(sheetNo, normalize):
    df = pd.read_excel(
        "/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week4/bayes/data.xlsx",
        sheetname=sheetNo)
    data = np.array(df.as_matrix())
    label = np.array(
        [1 if x == 1 else -1 for x in data[:, -1:]], dtype='int')
    data = np.array(data[:, :-1], dtype='float')
    num = len(label)
    featNum = len(data[0])
    # 特征归一化：
    if normalize:
        maxMat = np.row_stack((np.max(data, axis=0), ) * num)
        minMat = np.row_stack((np.min(data, axis=0), ) * num)
        data = (data - minMat) / (maxMat - minMat)
    return data, label, featNum, num


def chooseAlpha(trainData, trainLabel, num, b, alpha):
    '''# 废弃的代码（功能与下面的长表达式相同）：
    diffs = []
    for x, y in zip(trainData, trainLabel):
        pred = b + np.dot((trainLabel*alpha).transpose(), np.dot(trainData, x))
        diff = 1 - pred * y
        diffs.append(diff)'''
    diffs = [1-(b+np.dot((trainLabel*alpha).transpose(),
                         np.dot(trainData, x)))*y for x, y in zip(trainData, trainLabel)]
    i = diffs.index(max(diffs))    # 选择违背KKT条件最多的alpha值开始优化
    dist = [np.linalg.norm(x - trainData[i])
            for x, y in zip(trainData, trainLabel)]
    j = dist.index(max(dist))
    return i, j


def train(trainData, trainLabel, featNum, num):
    b = 0
    # alpha = np.zeros(num)
    alpha = np.random.randint(0, C, (num, ))
    i, j = chooseAlpha(trainData, trainLabel, num, b, alpha)
    c = -sum([trainLabel[k]*alpha[k] for k in range(num) if k!=i and k!=j])
    

def test(testData, testLabel, w, b):
    count = 0
    for x, y in zip(testData, testLabel):
        pred = np.dot(w, x)+b
        count += 1 if pred > 0 else 0
    return count/len(testLabel)


trainData, trainLabel, featNum, num = load(0, True)
train(trainData, trainLabel, featNum, num)
