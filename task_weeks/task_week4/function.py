import numpy as np
import pandas as pd


def load(sheetNo, normalize):
    df = pd.read_excel(
        "/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week4/bayes/data.xlsx",
        sheetname=sheetNo)
    data = np.array(df.as_matrix())
    label = np.array(
        [1 if x == 1 else -1 for x in data[:, -1:]], dtype='int')
    data = np.array(data[:, :-1], dtype='float')
    num = len(label)
    # 特征归一化：
    if normalize:
        maxMat = np.row_stack((np.max(data, axis=0), ) * num)
        minMat = np.row_stack((np.min(data, axis=0), ) * num)
        data = (data - minMat) / (maxMat - minMat)
    return data, label, num


def divide(trainData, trainLabel):
    newLabel = np.unique(trainLabel)
    newData = []
    for l in newLabel:
        newData.append(
            np.array([x for x, y in zip(trainData, trainLabel) if y == l]))
    return newData, newLabel


def test(testData, testLabel, func):
    count = 0
    for x, y in zip(testData, testLabel):
        prob = [f.cal(x) for f in func]
        ix = prob.index(max(prob))
        pred = func[ix].label
        if pred == y:
            count += 1
    return count/len(testLabel)
