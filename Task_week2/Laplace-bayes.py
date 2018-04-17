import numpy as np
import csv
test_num = 1


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

    data = np.array(data[1:])
    label = [(1 if x == '是' else 0) for x in data[:, -1:]]
    data = np.array(data[:, :-1], 'float32')

    train_data = data[:-test_num, :]
    train_label = label[:-test_num]
    test_data = data[-test_num:, :]
    test_label = label[-test_num:]
    return train_data, train_label, test_data, test_label


def train(train_data, train_label):
    p_c1 = train_label.count(1)/len(train_label)
    p_c0 = 1-p_c1
    num = len(train_label)
    length = len(train_data[0])
    p_x1 = []
    p_x0 = []
    tmp=[[],[]]
    for i in range(num):
        if train_label[i] == 1:
            tmp[1].append(train_data[i])
        else :
            tmp[0].append(train_data[i])

    for y in tmp:
        for j in range(length):
            column = [x[0] for x in (y[:, j:j+1]).tolist()]
            prob = {}
            if j < 6:
                feature = set(column)
                for x in feature:
                    prob[x] = column.count(x)/train_label.count(1)
                p_x1.append(prob)
            else:
                pass


train_data, train_label, test_data, test_label = load()
train(train_data, train_label)
