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
    new_label = []
    slice = data[:, uqFeat:uqFeat+1]
    data=np.delete(data,uqFeat,axis=1)
    values = np.unique(slice)
    for x in values:
        tmp_data = []
        tmp_label = []
        index = (np.where(slice == x))[0]
        for y in index:
            tmp_data.append(data[y])
            tmp_label.append(label[y])

        new_data.append(np.array(tmp_data))
        new_label.append(np.array(tmp_label))

    return new_data, new_label, values


# 寻找划分一个数据集的最佳方式(传入参数是一个训练集和其对应的标签)
def optimize(data, label):
    num = len(label)
    length = len(data[0])
    originEnt = cal_ent(label)
    maxGain = 0.0
    uqFeat = 0  # <-作为最佳划分依据的特征
    for i in range(length):
        slice = data[:, i:i+1]
        values = np.unique(slice)
        new_data, new_label = divide(data, label, i)
        sigma = 0
        for x in new_label:
            sigma += (cal_ent(x))*len(x)/num
        gain = originEnt-sigma
        if gain > maxGain:
            maxGain = gain
            uqFeat = i
    return uqFeat


def plant(data, label):
    values, counts = np.unique(label, return_counts=True)
    if counts == 1:
        return label[0]
    if np.shape(data[0])[0]==1:
        return values[np.argmax(counts)]
    tmp_data=data
    tmp_label=label
    uqFeat=optimize(tmp_data,tmp_label)
    tree={uqFeat:{}}
    new_data,new_label,featValue=divide(tmp_data,tmp_label,uqFeat)
    for i in np.shape(featValue):
        tree[uqFeat][featValue[i]]=plant(
            new_data[i],new_label[i])
    return tree


train_data, train_label = load('traindata.txt')
uqFeat = optimize(train_data, train_label)
