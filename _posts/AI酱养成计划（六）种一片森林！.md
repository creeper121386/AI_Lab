---
title: AI酱养成计划（六）种一片随机森林！
date: 2018-07-11 20:06:01
tags:
- 机器学习
- python
categories: 机器学习
---

![植树造林，从我做起](https://ss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=1842044919,4282973571&fm=27&gp=0.jpg)

***

<!--more-->

# $Bagging$集成学习 & 随机森林

## 集成学习

在机器学习的过程中，常常会遇到过拟合的问题。集成学习是解决过拟合问题的一种优良途径。所谓集成学习，指的是一次训练多个学习器，最后将这些学习器综合在一起进行使用。由于许多弱学习器便于训练，但效果并不理想，因此可以考虑通过弱学习器集成学习的办法，来获得一个强学习器来进行最终的分类或回归任务。有时，也会将不同类型的学习器进行集成（例如决策树和SVM）。

假设现有多个学习器$t_1,t_2,...t_n$，现在将这$n$个学习器集成学习，得到一个强学习器$T$。在具体的操作过程中，每个学习器$t_i$的训练过程和原来独立学习时大致相同，但要尽量满足各个学习器“好而不同”的原则——即在保证每个学习器分类/回归正确率的基础上，尽量使得学习器富有多样性（如果所有学习器都一样，那就和没集成一样了）。因此我们在训练子学习器$t_i$的过程中，常常希望引入一些随机因素，在我们要介绍的$Bagging$集成学习中，常用的方法是在划分训练集时使用一种随机采样法：自助采样法

### 自助采样

自助采样方法原理十分简单。假设现有的数据集是$\Bbb D$，我们想要从中划分出训练集$S$，自助采样的做法是：

* 首先令$S$为一空集
* 从$\Bbb D$中随机抽取一个样本$x$，加入到$S$中，$x$要放回$\Bbb D$中，以保证之后还可以被继续抽取
* 重复$|\Bbb D|$次抽取，即$\Bbb D$有多少元素，就随机抽取多少次
* 剔除$S$中的重复样本（集合中不能有相同元素）

根据采样过程，可以轻易地计算出来，当采样次数足够大时，假设$|\Bbb D|=m$，那么$\Bbb D$中任意一个元素不被抽取的概率是$1-\frac{1}{m}$，于是$S$中最终的元素个数是：

$$m\lim_{m \to \infty} \left( 1-\frac{1}{m} \right) ^m=\frac{1}{e} \approx 0.37m$$

因此，这样随机抽样的方法得到的训练集大小合适，剩下的部分可以作为验证集，或者用来辅助判断模型的过拟合情况（例如用来给决策树剪枝）。通过对每个子学习器进行自助采样，就可以保证每个子学习器是从不同的数据集训练出来的，也就保证了学习器的多样性。

### 模型测试

集成模型的测试过程与单一模型不同。由于包含多个子模型，在进行预测时$n$个子学习器会给出$n$个结果。

* 对于分类任务，往往直接选取出现次数最多的类别最为预测结果
* 对于回归任务，采用求均值的方法：
    * 可以直接求算数平均数$y=\sum_{i=1}^ny_i$
    * 也可以采用加权均值$y=\sum_{i=1}^nw_iy_i$，权重也作为待学习的参数（例如可以采用梯度下降的方法）。

接下来，我们以随机森林算法为例，实现$bagging$集成学习。

## 随机森林

随机森林顾名思义，是由很多棵决策树集成的森林。(什么是`决策树`？请看我的[这篇博客](https://creeper121386.github.io/2018/04/26/AI%E9%85%B1%E5%85%BB%E6%88%90%E8%AE%A1%E5%88%92%EF%BC%88%E4%B8%80%EF%BC%89%E7%A7%8D%E4%B8%80%E6%A3%B5%E5%86%B3%E7%AD%96%E6%A0%91%EF%BC%81/)）在进行训练时，单个决策树的训练、剪枝等方法与之前相同，要做出的一些改变是：

* 训练集的划分采用自助采样法，划分剩下的数据可以用来剪枝。
* 在决策树分支时，以往的做法是，计算出当前节点处按各个特征划分时的信息增益，并选出增益最大的特征，以该特征为依据进行数据划分。现在我们有了多个决策树，我们希望为每棵树引入一些随机量，来保证集成学习的多样性。因此，在当前节点可供划分的$p$个特征中，随机地选出$k$个计算信息熵（一般来说，选择$k={log_2}^p$效果较好），并在这$k$个特征中选出信息增益最大的一个进行划分。

### 测试

接下来给出随机森林的python代码。首先构建一个类封装随机森林的树结构、数据集以及训练和测试方法，其中数据集使用自助采样的方法：

```python
class Forest(object):
    def __init__(self, trainData, testData, trainLabel, testLabel):
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.testData = testData
        self.testLabel = testLabel
        self.trainIndex = []
        self.trees = []
        for _ in range(treeNum):
            self.trees.append([])
            self.trainIndex.append(set([np.random.randint(0, dataNum)
                                        for _ in range(dataNum)]))
```

接下来调用之前在[这篇博客](https://creeper121386.github.io/2018/04/26/AI%E9%85%B1%E5%85%BB%E6%88%90%E8%AE%A1%E5%88%92%EF%BC%88%E4%B8%80%EF%BC%89%E7%A7%8D%E4%B8%80%E6%A3%B5%E5%86%B3%E7%AD%96%E6%A0%91%EF%BC%81/)中写好的`API`，构造最简单的$ID3$决策树，并在节点划分的部分加以修改，封装`train()`和`test()`方法：

```python
    def train(self):
        for i in range(treeNum):
            featNO = [0, 1, 2, 3, 4, 5]
            data = np.array([self.trainData[x] for x in self.trainIndex[i]])
            label = np.array([self.trainLabel[x] for x in self.trainIndex[i]])
            self.trees[i] = API.plant(data, label, featNO)

    def test(self):
        count = 0
        num = len(self.testLabel)
        predLabel = np.array([API.pred(self.testData, T) for T in self.trees])
        finalPred = []
        for i in range(num):
            pred = [predLabel[j][i][1] for j in range(treeNum)]
            finalPred.append(max(pred ,key=pred.count))      
        for i in range(num):
            if self.testLabel[i] == finalPred[i]:
                count += 1
        acc = count/num
        return acc
```

这样就完成了一个随机森林。经过测试，之前的决策树正确率在$0.87$左右，而一个包含$20$棵树的随机森林，在相同的测试集上，正确率平均可以达到$0.985$，可见集成学习的威力。让我们一起植树造林吧！ヽ(✿ﾟ▽ﾟ)ノ

***

$2018.7.11 \;\; by \; WHY$