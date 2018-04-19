import numpy as np
import csv
import matplotlib.pyplot as plt
N = 5000
k = 3


def load():
    with open("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/hr-analytics.csv", "r") as f:
        reader = csv.reader(f)
        data = []
        i = 0
        for x in reader:
            data.append(x)
            if i == N:
                break
            i += 1

        del(data[0])
    num = len(data)
    label = []
    feature = []
    value = []
    for i in range(num):
        label.append(data[i][-1])
        data[i] = data[i][1:-2]
        tmp = data[i][-1]
        no = 0
        if tmp not in feature:
            feature.append(tmp)
            value.append(no)
            data[i][-1] = no
            no += 1
        else:
            data[i][-1] = value[feature.index(tmp)]
    data = np.array(data, dtype="float32")
    label = np.array(label, "int")
    return data, label


def cal_JC(category, cate_label, label, k):
    index = []
    num = len(label)
    est_label = [0]*num
    for x in cate_label:
        index.append(max(x, key=x.count))
    for i in range(k):
        for y in category[i]:
            for j in range(num):
                if (data[j]).all()==y.all():
                    ix=j
                    break
            est_label[ix] = index[i]
    a = b = c = 0
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            if label[i] == label[j]:
                if est_label[i] == est_label[j]:
                    a += 1
                else:
                    b += 1
            elif est_label[i] == est_label[j]:
                c += 1
    JC = a/(a+b+c)
    return JC


def divide(data, label, k):
    num = len(data)
    length = len(data[0])
    ave = np.ones((k, length))
    # ave=[0]*k
    category = []
    cate_label = []
    for i in range(k):
        category.append([])
        cate_label.append([])
        ix = np.random.choice(range(num), replace=False)
        ave[i] = data[ix]

    for j in range(50):
        for i in range(k):
            category[i] = []
            cate_label[i] = []
        for i in range(num):
            d = [np.linalg.norm(data[i] - y) for y in ave]
            ix = d.index(min(d))
            category[ix].append(data[i])
            cate_label[ix].append(label[i])
        new_ave = np.ones((k, length))
        # new_ave=[0]*k
        for i in range(k):
            new_ave[i] = np.mean(category[i], axis=0)
            '''if (new_ave[i]).all() != (ave[i]).all():
                ave[i]=new_ave[i]'''
        tmp = np.mean(new_ave-ave)
        if np.fabs(tmp) < 1e-60:
            break
        else:
            ave = new_ave
        JC = cal_JC(category, cate_label, label, k)
    print("JC =", JC)
    return ave, category, cate_label


def paint(category, ave, k):
    color = ['r', 'g', 'y', 'b', 'k', 'm', 'c']
    x = 3
    for i in range(k):
        plt.scatter(ave[i][x], ave[i][x+1], c='k', alpha=1)
        for y in category[i]:
            plt.scatter(y[x], y[x+1], c=color[i], alpha=0.4)
            # plt.plot(ave[i],y,c='b')
    plt.show()


data, label = load()
ave, category, cate_label = divide(data, label, k)
#paint(category, ave, k)
