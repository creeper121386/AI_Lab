import numpy as np
import csv
import matplotlib.pyplot as plt
# k=2


def load():
    with open("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon4.0.csv", "r") as f:
        reader = csv.reader(f)
        data = [x for x in reader]
    data = np.array(data, dtype="float64")
    return data


def divide(data, k):
    num = len(data)
    ave = np.ones((k, 2))
    # ave=[0]*k
    category = []
    for i in range(k):
        category.append([])
        ix = np.random.choice(range(num), replace=False)
        ave[i] = data[ix]
    for j in range(100):
        for i in range(k):
            category[i] = []
        for x in data:
            d = [np.linalg.norm(x - y) for y in ave]
            category[d.index(min(d))].append(x)
        new_ave = np.ones((k, 2))
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
    return ave, category


def paint(category, ave, k):
    color = ['r', 'g', 'y', 'b', 'k', 'm', 'c']
    for i in range(k):
        plt.scatter(ave[i][0], ave[i][1], c='b')
        for y in category[i]:
            plt.scatter(y[0], y[1], c=color[i])
            # plt.plot(ave[i],y,c='b')
    plt.show()


def cal_DBI(category, ave, k):
    avg = []
    DBI = 0
    for ix in range(k):
        num = len(category[ix])
        sum_dist = 0
        for x in category[ix]:
            for y in category[ix]:
                sum_dist += np.fabs(np.linalg.norm(x - y))
        avg.append((sum_dist)/(num*(num-1)))
    for i in range(k):
        DBI += max([(avg[i]+avg[j])/np.fabs((np.linalg.norm(ave[i]-ave[j])))
                    for j in range(k) if j != i])
    DBI /= k
    return DBI


data = load()
DBI = []
for k in range(2, 6):
    ave, category = divide(data, k)
    if min([len(x) for x in category]) <= 1:
        continue
    DBI.append(cal_DBI(category, ave, k))
k = DBI.index(min(DBI))
paint(category, ave, k)
