import numpy as np
import csv
from math import log
N = 350

# 计算信息熵(需要单独传入标签):


def cal_ent(label):
    size = len(label)
    num1 = 0  # 分别表示类别0和1的数目
    num0 = 0
    for x in label:
        if x == 0:
            num0 += 1
        else:
            num1 += 1
    p0 = float(num0)/size
    p1 = float(num1)/size
    if p0*p1 != 0:
        return -p0*log(p0, 2)-p1*log(p1, 2)
    elif p0 == 0 or p1 == 0:
        return 0


# 划分数据集:数据集包括特征值和标签
def divide(data, label, feature_no, mid_value):
    data1 = []
    data2 = []
    label1 = []
    label2 = []
    for i in range(len(label)):
        tmp = data[i][:feature_no]+data[i][feature_no:]
        if data[i][feature_no] >= mid_value:
            data1.append(tmp)
            label1.append(label[i])
        elif data[i][feature_no] < mid_value:
            data2.append(tmp)
            label2.append(label[i])
    return data1, data2, label1, label2


# 寻找最佳的划分方式:
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
            _, _, label1, label2 = divide(data, label, i, mid_value)
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


def plant_tree(data, label):
    if len(set(label)) == 1:
        return label[0]
    if len(data[0]) == 1:
        return (1 if label.count(1) > label.count(0) else 0)
    tmp_label = label
    feature, mid_value = optimize(data, tmp_label)
    tree = {(feature, mid_value): {}}
    data1, data2, label1, label2 = divide(data, label, feature, mid_value)
    tree[(feature,mid_value)]['more'] = plant_tree(data1, label1)
    tree[(feature,mid_value)]['less'] = plant_tree(data2, label2)
    return tree


with open("pima-indians-diabetes.data.csv", "r") as f:
    reader = csv.reader(f)
    data = [x for x in reader]
'''data = np.array(data, dtype="float64")
label = data[:, -1:]
data = data[:, 0:-1]'''
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])

label = [x[-1] for x in data]
data = [x[0:-1] for x in data]
train_data = data[:N]
test_data = data[N:]
train_label = label[:N]
test_label = label[N:]

tree = plant_tree(train_data, train_label)
print(tree)