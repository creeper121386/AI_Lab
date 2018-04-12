import numpy as np
import csv
N = 350


def cal_ent(data, label):
    num = len(data)
    label_num = [0, 0]  # 分别表示类别为0和1的概率
    for x in label:
        if x == 0:
            label_num[0] += 1
        else:
            label_num[1] += 1
    p0 = label_num[0]/num
    p1 = label_num[0]/num
    ent = -p0-p1
    return ent


with open("pima-indians-diabetes.data.csv", "r") as f:
    reader = csv.reader(f)
    data = [x for x in reader]
data = np.array(data, dtype="float64")
label = data[:, -1:]
data = data[:, 0:-1]
ent = cal_ent(data, label)
print(ent)


train_data = data[:N, :]
test_data = data[N:, :]
