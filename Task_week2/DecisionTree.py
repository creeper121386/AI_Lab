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


def optimize(data, label):
    length = len(data)
    feature_num = len(data[0])
    originEnt = cal_ent(label)
    feature = 0
    maxGain = 0.0
    special_value = 0
    for i in range(feature_num):
        value = [x[i] for x in data]
        value = set(value)  # 一开始没想到用set...妙哇...
        for mid_value in value:
            new_data = divide(data, label, i)
            len1 = len(label1)
            len2 = len(label2)
            if len1*len2 != 0:
                ent1 = cal_ent(label1)
                ent2 = cal_ent(label2)
                gain = originEnt-(ent1*len1+ent2*len2)/length
            else:
                continue
            #print('%d %f %f %f'%(i, mid_value,ent1,ent2))
            if gain > maxGain:
                maxGain = gain
                feature = i
                special_value = mid_value
    return feature, special_value

train_data, train_label = load('traindata.txt')

