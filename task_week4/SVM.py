import pandas as pd
import numpy as np
loopTimes = 10  # 更新第二个alpha变量的循环次数
kernel = 0  # 使用哪个核(0代表不使用，核函数编号见列表K)
theta = -1  # sigmoid核参数
beta = 1    # sigmoid核参数
d = 3   # 多项式核的指数
C = 1e5
sigma = 2
K = [None, liner, multi, Gauss, Laplace, sigmoid]


def liner(x, y):
    return np.dot(x, y)


def multi(x, y):
    return (np.dot(x, y))**d


def Gauss(x, y):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))


def Laplace(x, y):
    return np.exp(-(np.linalg.norm(x-y))/sigma)


def sigmoid(x, y):
    return np.tanh(beta*np.dot(x, y)+theta)


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


def pred(feat, b, alpha, kernel, trainData=trainData, trainLabel=trainLabel):
    if kernel:
        return b + np.dot((trainLabel*alpha).transpose(), np.array([K[kernel](x, feat) for x in trainData]))
    else:
        return b + np.dot((trainLabel*alpha).transpose(), np.dot(trainData, feat))


def chooseAlpha(trainData, trainLabel, num, b, alpha, index):
    '''# 废弃的代码（功能与下面的长表达式相同）：
    E = []
    for x, y in zip(trainData, trainLabel):
        pred = b + np.dot((trainLabel*alpha).transpose(), np.dot(trainData, x))
        diff = 1 - pred * y
        E.append(diff)'''
    E = [1-pred(trainData[i], b, alpha, kernel)*trainLabel[i] for i in index]
    Ei = max(E)
    i = index[E.index(Ei)]    # 选择违背KKT条件最多的alpha值开始优化
    j = np.argmax(np.abs(np.array(E) - Ei))
    Ej = E[j]
    j = index[j]
    return i, j, Ei, Ej


def train(trainData, trainLabel, featNum, num):
    b = 0
    # alpha = np.zeros(num)
    alpha = np.random.randint(0, C, (num, ))
    index = range(num)
    while len(index):
        i, j, Ei, Ej = chooseAlpha(trainData, trainLabel, num, b, alpha, index)
        preI = alpha[i].copy()
        preJ = alpha[j].copy()
        if trainLabel[i] == trainLabel[j]:
            L = max(0, alpha[j]+alpha[i]-C)
            H = min(C, C+alpha[j]-alpha[i])
        else:
            L = max(0, alpha[j]-alpha[i])
            H = min(C, alpha[j]+alpha[i])
        # c = -sum([trainLabel[k]*alpha[k]
        #          for k in range(num) if k != i and k != j])
        eta = 2 * np.dot(trainData[i], trainData[j].transpose()) - \
            np.dot(trainData[i], trainData[i].transpose()) - \
            np.dot(trainData[j], trainData[j].transpose())
        if L != H and eta <= 0:
            alpha[j] -= trainLabel[j] * (Ei-Ej)/eta
            if alpha[j]>H:  alpha[j] = H
            if alpha[j]<L:  alpha[j] = L
            if np.abs(alpha[j]-preJ)<1e-3:
                continue
            alpha[i] += trainLabel[i]*trainLabel[j]*(preJ-alpha[j])



def test(testData, testLabel, w, b):
    count = 0
    for x, y in zip(testData, testLabel):
        pred = np.dot(w, x)+b
        count += 1 if y*pred >= 1 else 0
    return count/len(testLabel)


trainData, trainLabel, featNum, num = load(0, True)
train(trainData, trainLabel, featNum, num)
