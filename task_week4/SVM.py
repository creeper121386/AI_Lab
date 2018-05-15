import numpy as np
import pandas as pd
C = 1e5
loopTimes = 100

def load(sheetNo, normalize):
    df = pd.read_excel(
        "/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week4/bayes/data.xlsx",
        sheetname=sheetNo)
    data = np.array(df.as_matrix())
    data = np.array(data[:, :-1], dtype='float')
    label = np.array(
        [1 if x == 1 else -1 for x in data[:, -1:]], dtype='int')
    num = len(label)
    featNum=len(data[0])
    # 特征归一化：
    if normalize:
        maxMat = np.row_stack((np.max(data, axis=0), ) * num)
        minMat = np.row_stack((np.min(data, axis=0), ) * num)
        data = (data - minMat) / (maxMat - minMat)
    return data, label, featNum, num

def chooseAlpha():

def train(trainData, trainLabel, featNum, num):
    b = 0
    alpha = np.zeros(featNum)
    # alpha = np.random.randn(0, C, (1, featNum))
    for x, y in zip(trainData, trainLabel):
        for x in 
        

def test(testData, testLabel, w, b):
    count = 0
    for x, y in zip(testData, testLabel):
        pred = np.dot(w, x)+b
        count += 1 if pred > 0 else 0
    return count/len(testLabel)


trainData, trainLabel, featNum, num = load(0, True)
