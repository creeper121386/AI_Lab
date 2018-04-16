import numpy as np
import csv


def load():
    with open("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon.csv", "r") as f:
        reader = csv.reader(f)
        data = [x[1:] for x in reader]
    data = np.array(data[1:])
    label = [(1 if x == '是' else 0) for x in data[:, -1:]]
    tmp = np.array(data[:, -3:-1], 'float32')
    chinese=data[:,:-3]
    print(chinese[0][0])
    
    for i in range(len(chinese[0])):
        feature={}
        j=0
        for x in chinese[:,i:i+1]:
            if x not in feature:
                feature[x]=j
                j+=1
        for j in range(len(chinese)):
            chinese[j][i]=feature[chinese[j][i]]


    train_data = np.column_stack((data[:, 0:-3], tmp))
    print(type(train_data[0][-1]))
    return train_data, train_label


def train(train_data, train_label):
    p_c1 = train_label.count(1)/len(train_label)
    p_c2 = 1-p_c1


train_data, train_label = load()
