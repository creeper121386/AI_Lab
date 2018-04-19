import numpy as np


def load(f_name):
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/tree/" +
                      f_name, dtype=np.str, delimiter=' ')
    data = data[1:, :]
    label = data[:, -1:]
    data = data[:, :-1]
    return data, label


def cal_ent(label):
    num = len(label)
    _, counts = np.unique(label, return_counts=True)
    prob = counts/num
    ent = -(np.sum(prob*np.log2(prob)))
    return ent


def divide(data, label, uqFeat):
    new_data = []
    slice = data[:, uqFeat:uqFeat+1]
    values = np.unique(slice)
    for x in values:
        index = (np.where(slice == x))[0]
        for y in index:
            new_data.append(data[y])
    return new_data


train_data, train_label = load('traindata.txt')
ent = cal_ent(train_label)
