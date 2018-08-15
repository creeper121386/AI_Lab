import numpy as np
import csv
test_num = 1
import math

def load():
    with open("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon.csv", "r") as f:
        reader = csv.reader(f)
        data = [x[1:] for x in reader]
    data = data[1:]
    for i in range(len(data[0])-3):
        feature = []
        value = []
        num = 0
        for j in range(len(data)):
            tmp = data[j][i]
            if tmp not in feature:
                feature.append(tmp)
                value.append(num)
                data[j][i] = num
                num += 1
            else:
                data[j][i] = value[feature.index(tmp)]
    train_data = data[:-1]
    test_data = data[-1:]
    test_data = test_data[0][:-1]
    pos = []
    neg = []
    for i in range(len(train_data)):
        if data[i][-1] == '是':
            pos.append(data[i])
        else:
            neg.append(data[i])

    pos = np.array(pos)
    neg = np.array(neg)
    pos = pos[:, :-1]
    neg = neg[:, :-1]
    pos = np.array(pos, 'float32')
    neg = np.array(neg, 'float32')
    '''
    data = np.array(data[1:])
    label = [(1 if x == '是' else 0) for x in data[:, -1:]]
    train_data = data[:-test_num, :]
    train_label = label[:-test_num]
    test_data = data[-test_num:, :]
    test_label = label[-test_num:]
    '''
    return pos, neg, test_data


def train(pos, neg):
    num = [len(neg), len(pos)]
    length = len(pos[0])
    pc = [(num[0]+1)/(num[1]+num[0]+2), (num[1]+1)/(num[1]+num[0]+2)]  # 使用拉普拉斯修正
    feat = [[], []]
    p_xc = [[], []]
    i = 0
    for y in (neg, pos):
        for j in range(length):
            column = y[:, j:j+1]
            if j < 6:
                feature, prob = np.unique(column, return_counts=True)
                prob = (prob+1)/(num[i]+len(feature))
                p_xc[i].append(prob)
                feat[i].append(feature)
            else:
                ave = np.mean(column)
                sigma = np.var(column)
                p_xc[i].append((ave, sigma))
        i += 1
    return feat, p_xc, pc


def test(test_data, feat, p_xc, pc):
    p_pred = pc
    test_data=np.array(test_data,'float32')
    for x in (0, 1):
        for i in range(len(test_data)):
            if i < 6:
                #ix=(feat[x][i]).index(test_data[i])
                ix=np.where(feat[x][i]==test_data[i])
                p_pred[x]*=p_xc[x][i][ix]
            else:
                ave,sigma=p_xc[x][i]
                p=np.exp(-(test_data[i]-ave)**2/(2*sigma**2))/(np.sqrt(2*3.14)*sigma)
                p_pred[x]*=p
    if p_pred[0]>p_pred[1]:
        return 0
    else:
        return 1

pos, neg, test_data = load()
feat, p_xc, pc = train(pos, neg)
pre=test(test_data, feat, p_xc, pc)
if pre==1:
    print("Right!")
else:
    print("Wrong!")