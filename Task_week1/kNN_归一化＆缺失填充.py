import numpy as np
import matplotlib as plt
import csv

N = 350
k = 20


def classify(train_data, train_label, test_data):
    result = []
    for y in test_data:
        distance = []
        i = 0
        for x in train_data:
            #tmp = np.sqrt(np.sum(np.square(y-x)))
            tmp = np.linalg.norm(y-x)
            distance.append((tmp, i))
            i += 1
        distance.sort(key=lambda member: member[0])
        pos_num = neg_num = 0
        for i in range(k):
            index = distance[i][1]
            if train_label[index] == 1:
                pos_num += 1
            elif train_label[index] == 0:
                neg_num += 1
        result.append(1 if pos_num > neg_num else 0)
    result = np.array(result, "int")
    return result


def fill(data, length):
    # 处理缺失值：
    for j in range(8):
        tmp = num = 0
        for i in range(length):
            if data[i][j]:
                tmp += data[i][j]
                num += 1
        ave = tmp/num
        for i in range(length):
            if not data[i][j]:
                data[i][j] = tmp/ave
        return data


def normalize(data, length):
    # 数据归一化处理：
    Max = np.max(data, axis=0)
    Min = np.min(data, axis=0)
    for j in range(8):
        for i in range(length):
            data[i][j] = (data[i][j]-Min[j])/(Max[j]-Min[j])
    return data


with open("pima-indians-diabetes.data.csv", "r") as f:
    reader = csv.reader(f)
    print()
    data = [x for x in reader]
data = np.array(data, dtype="float64")
length = len(data)

data = fill(data, length)
data = normalize(data, length)

# 划分测试集和训练集:
train_data = data[:N, :]
test_data = data[N:, :]
train_label = train_data[:, -1:]
train_data = train_data[:, 0:-1]
test_label = test_data[:, -1:]
test_data = test_data[:, 0:-1]
# 不对数据进行处理和填补:
result = classify(train_data, train_label, test_data)
error = 0
num = len(test_label)
for i in range(num):
    if test_label[i] != result[i]:
        error += 1
print(error/num)
