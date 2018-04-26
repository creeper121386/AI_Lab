import numpy as np
import matplotlib.pyplot as plt
#import draw_Tree as draw
k = 10
leafNo = -1
newLeafNo = -1


def load(f_name, divide, delimiter):
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/tree/" +
                      f_name, dtype=np.str, delimiter=delimiter)
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

    # 调试中发现,如果进行预剪枝,直接一刀就剪没了.....将会导致决策树层数很浅,严重欠拟合,因此不作预剪枝步骤,以下代码仅作记录:
    # 开始预剪枝:若不对结点进行划分,该节点对应的叶节点为:
    '''values, counts = np.unique(label, return_counts=True)
    maxValue = values[np.argmax(counts)]
    values, counts = np.unique(valiLabel, return_counts=True)
    acc1 = counts[np.where(values == maxValue)[0]]/len(valiLabel)

    _, new_label, featValue = divide(data, label, uqFeat)
    n = len(featValue)
    tmpLabel = []
    for i in range(n):
        values, counts = np.unique(new_label[i], return_counts=True)
        value = values[np.argmax(counts)]
        tmpLabel.append(value)
    count = 0
    for i in range(len(valiLabel)):
        for j in range(n):
            if featValue[j] == valiData[i][uqFeat] and tmpLabel[j] == valiLabel[i]:
                count += 1
    acc2 = count/len(valiLabel)
    if acc1 <= acc2:
        return uqFeat
    else:
        return maxValue'''
    return uqFeat


def plant(data, label, feat_label, valiData, valiLabel):
    values, counts = np.unique(label, return_counts=True)
    if np.shape(values)[0] == 1:
        global leafNo
        leafNo += 1
        return (leafNo, label[0][0])
    if np.shape(data[0])[0] == 1:
        leafNo += 1
        return (leafNo, values[np.argmax(counts)])
    uqFeat = optimize(data, label, valiData, valiLabel)
    if type(uqFeat).__name__ != 'int':
        leafNo += 1
        return (leafNo, uqFeat)

    uqFeatLabel = feat_label[uqFeat]
    tree = {uqFeatLabel: {}}
    del(feat_label[uqFeat])
    new_data, new_label, featValue = divide(data, label, uqFeat)
    for i in range(np.shape(featValue)[0]):
        subLabel = feat_label[:]
        tree[uqFeatLabel][featValue[i]] = plant(
            new_data[i], new_label[i], subLabel, valiData, valiLabel)
    return tree


def classify(data, tree):
    for x in tree.keys():
        ix = x
    dict = tree[ix]
    for key in dict.keys():
        if data[ix] == key:
            if type(dict[key]).__name__ == 'dict':
                label = classify(data, dict[key])
            else:
                label = dict[key]
            return label


def pred(data, tree):
    pred_label = []
    i = 0
    for x in data:
        i += 1
        label = classify(x, tree)
        if label != None:
            pred_label.append(label)
        else:
            pred_label.append((-1, 'uacc'))
    return pred_label


def test(data, tree, label):
    pred_label = pred(data, tree)
    num = len(label)
    count = 0
    for i in range(num):
        if label[i] == pred_label[i][1]:
            count += 1
    acc = count/num
    return acc


def cut(tree, data, label):
    global newLeafNo
    if len(label) == 0:
        return tree
    for x in tree.keys():
        ix = x
    dict = tree[ix]
    for key in dict.keys():
        tmpData = []
        tmpLabel = []
        for i in range(len(data)):
            if data[i][ix] == key:
                tmpData.append(data[i])
                tmpLabel.append(label[i])
        tmpData = np.array(tmpData)
        tmpLabel = np.array(tmpLabel)
        if type(dict[key]).__name__ == 'dict':
            new_tree = cut(dict[key], tmpData, tmpLabel)
            tree[ix][key] = new_tree
        else:
            value, count = np.unique(label, return_counts=True)
            tmp = np.argmax(count)
            newAcc = count[tmp]/len(label)
            oldAcc = test(data, tree, label)
            if newAcc > oldAcc:
                tree = (newLeafNo, value[tmp])
                newLeafNo -= 1
                return tree
    return tree


def cal_ROC(predLabel, label):
    #label = np.array(label, dtype='int')
    #label = label.tolist()
    #label = [x[0] for x in label]
    no1 = label.count(1)+1e-8
    no0 = label.count(0)+1e-8
    FP = TP = 0
    for i in range(len(label)):
        if label[i] == predLabel[i] == 1:
            TP += 1
        elif label[i] == 0 and predLabel[i] == 1:
            FP += 1
    return TP/no1, FP/no0


def cal_PR(predLabel, label):
    no_real = label.count(1)+1e-8
    no_pred = predLabel.count(1)+1e-8
    TP = 0
    for i in range(len(label)):
        if label[i] == predLabel[i] == 1:
            TP += 1
    return TP/no_pred, TP/no_real


def draw_ROC(data, test_label, tree, leafNum):
    values = np.unique(test_label)
    num = len(test_label)
    k = 0
    for x in values:
        label = []
        pred_label = pred(data, tree)
        pred_leaf = []
        for i in range(num):
            label.append(1 if test_label[i] == x else 0)
            pred_leaf.append(pred_label[i][0])
        uqLeaf = np.unique(pred_leaf)
        posAcc = []
        for l in uqLeaf:
            index = np.where(pred_leaf == l)[0]
            posNum = 0
            for ix in index:
                if label[ix] == 1:
                    posNum += 1
            posAcc.append([l, posNum/len(index)])
        maxLeaf = sorted(posAcc, key=lambda temp: temp[1], reverse=True)
        maxLeaf = [y[0] for y in maxLeaf]
        leaves = [0]*(leafNum+1)    # 按照排序后的叶节点顺序,表示该分类器下,对应所有叶节点的分类情况
        x_ROC = []
        y_ROC = []
        x_PR = []
        y_PR = []
        for i in range(leafNum+1):
            leaves[i] = 1
            newPredLabel = []
            for j in range(num):
                tmp = pred_leaf[j]
                ix = maxLeaf.index(tmp)
                newPredLabel.append(leaves[ix])
            Tp, Fp = cal_ROC(newPredLabel, label)
            p1, p2 = cal_PR(newPredLabel, label)
            x_ROC.append(Fp)
            y_ROC.append(Tp)
            x_PR.append(p1)
            y_PR.append(p2)
        fig = plt.figure(k)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('False Postive Rate')
        ax.set_ylabel('True Postive Rate')
        ax.set_title('ROC Curve of Decision Tree (pre-pruning)')
        plt.plot(x_ROC, y_ROC)
        plt.scatter(x_ROC, y_ROC, alpha=0.6)
        plt.show()

        '''ax.set_xlabel('P')
        ax.set_ylabel('R')
        ax.set_title('PR Curve of Decision Tree (pre-pruning)')
        plt.plot(x_PR, y_PR)
        plt.scatter(x_PR, y_PR, alpha=0.6)
        plt.show()
        k += 1'''


data, label = load('traindata.txt', divide=True, delimiter=' ')
num = len(label)
trees = []
accs = []
for i in range(k):
    ix = int(i+num/k)
    valiData = data[i:ix, :]
    valiLabel = label[i:ix, :]
    train_data = np.vstack((data[:i, :], data[ix:, :]))
    train_label = np.vstack((label[:i, :], label[ix:, :]))

    feat_label = [0, 1, 2, 3, 4, 5]
    # feat=optimize(train_data,train_label,valiData,valiLabel)
    leafNo = -1
    tree = plant(train_data, train_label, feat_label, valiData, valiLabel)

    acc = test(valiData, tree, valiLabel)
    print('acc =', acc)
    accs.append(acc)
    trees.append(tree)


tree = trees[accs.index(max(accs))]
newTree = cut(tree, train_data, train_label)
draw_ROC(train_data, train_label, newTree, leafNo+1)
acc = test(valiData, newTree, valiLabel)
print(acc)

test_data = load('testdata.txt', divide=False, delimiter=' ')
pred_label = pred(test_data, tree)
print(pred_label)

'''tmpData,tmpLabel=load('tmp.txt',divide=True,delimiter=',')
acc = test(tmpData, tree, tmpLabel)
print(acc)'''
