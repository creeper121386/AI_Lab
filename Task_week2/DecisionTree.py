import numpy as np
import matplotlib.pyplot as plt
#import draw_Tree as draw
k=10

def load(f_name, divide):
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/tree/" +
                      f_name, dtype=np.str, delimiter=' ')
    data = data[1:, :]
    if divide:
        label = data[:, -1:]
        data = data[:, :-1]
        return data, label
    else:
        return data


def cal_ent(label):
    num = len(label)
    _, counts = np.unique(label, return_counts=True)
    prob = counts/num
    ent = (np.sum(prob*np.log2(prob)))
    return -ent


def divide(data, label, uqFeat):
    new_data = []
    new_label = []
    slice = data[:, uqFeat:uqFeat+1]
    data = np.delete(data, uqFeat, axis=1)
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


# 寻找划分一个数据集的最佳方式(传入参数是一个数据集和其对应的标签)
def optimize(data, label, valiData, valiLabel):
    num = len(label)
    length = len(data[0])
    originEnt = cal_ent(label)
    maxGain = 0.0
    uqFeat = 0  # <-作为最佳划分依据的特征
    for i in range(length):
        _, new_label, _ = divide(data, label, i)
        sigma = 0
        for x in new_label:
            sigma += (cal_ent(x))*len(x)/num
        gain = originEnt-sigma
        if gain > maxGain:
            maxGain = gain
            uqFeat = i
    # 若不对结点进行划分,该节点对应的叶节点为:
    values, counts = np.unique(label, return_counts=True)
    value=values[np.argmax(counts)]
    values, counts = np.unique(valiLabel, return_counts=True)
    acc1=counts[np.where(values==value)[0]]/len(valiLabel)

    _,new_label,featValue=divide(data,label,uqFeat)
    n=len(featValue)
    tmpLabel=[]
    for i in range(n):
        values, counts = np.unique(new_label[i], return_counts=True)
        value = values[np.argmax(counts)]
        tmpLabel.append(value)

    count=0
    for i in range(len(valiLabel)):
        for j in range(n):
            if featValue[j] == valiData[i][uqFeat] and tmpLabel[j] == valiLabel[i]:
               count+=1
    acc2=count/len(valiLabel)
    if acc1<acc2:
        return uqFeat
    else:
        return -1


def plant(data, label,feat_label,valiData,valiLabel):
    values, counts = np.unique(label, return_counts=True)
    if np.shape(values)[0] == 1:
        return label[0][0]
    if np.shape(data[0])[0] == 1:
        return values[np.argmax(counts)]
    uqFeat = optimize(data, label,valiData,valiLabel)
    uqFeatLabel=feat_label[uqFeat]
    tree = {uqFeatLabel: {}}
    del(feat_label[uqFeat])
    new_data, new_label, featValue = divide(data, label, uqFeat)
    for i in range(np.shape(featValue)[0]):
        subLabel=feat_label[:]
        tree[uqFeatLabel][featValue[i]] = plant(
            new_data[i], new_label[i], subLabel,valiData,valiLabel)
    return tree


def classify(data,tree):
    for x in tree.keys():
        ix=x
    dict=tree[ix]
    for key in dict.keys():
        if data[ix]==key:
            if type(dict[key]).__name__ == 'dict':
                label=classify(data,dict[key])
            else:
                label=dict[key]
            return label
    
def pred(data, tree):
    pred_label = []
    for x in data:
        label = classify(x,tree)
        pred_label.append(label)
    return pred_label

def test(data,tree,label):
    pred_label=pred(data,tree)
    num = len(label)
    count=0
    for i in range(num):
        if label[i]==pred_label[i]:
            count+=1
    acc=count/num
    return acc


data, label = load('traindata.txt', divide=True)
num=len(label)
trees=[]
accs=[]
for i in range(k):
    ix=int(i+num/k)
    valiData=data[i:ix,:]
    valiLabel=label[i:ix,:]
    train_data=np.vstack((data[:i,:],data[ix:,:]))
    train_label = np.vstack((label[:i, :],label[ix:,:]))

    feat_label = [0, 1, 2, 3, 4, 5]
    tree = plant(train_data, train_label,feat_label,valiData,valiLabel)
    acc=test(valiData,tree,valiLabel)
    accs.append(acc)
    trees.append(tree)

tree=trees[accs.index(max(accs))]
test_data = load('testdata.txt', divide=False)
pred_label=pred(test_data,tree)
