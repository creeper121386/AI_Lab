import numpy as np
import csv

N = 350
k = 10


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


with open("pima-indians-diabetes.data.csv", "r") as f:
    reader = csv.reader(f)
    data = [x for x in reader]
data = np.array(data, dtype="float64")
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
