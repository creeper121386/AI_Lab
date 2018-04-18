---
title: why的numpy学习记录！
date: 2018-04-11 19:10:51
tags:
- 编程
categories: python

---

# why的nmupy学习记录！

#### 导入：`import np as np`
##　数据类型
**数组（array）：** 建立数组：np.array(列表)，将列表变成array。

***

## API
* `np.zeros((m,n))`：建立一个元素均为0的m*n矩阵。
* `np.ones((m,n))`：建立一个元素都是1的m*n矩阵。
* `np.eye(m)`：建立一个m阶对角线方阵，主对角线元素均为1。
* `np.full((m,n),p)`：建立一个元素均为p的m*n矩阵。
* `np.random.random((m,n))`：建立一个m*n的随机数值的矩阵。

***

## 矩阵操作
* **切片：** array本质上是多维数组，因此对每一个子数组均需要指定切片。如`a[1:2,5:6]`表示将a矩阵的1~2行，5~6行切片。切片时`:`表示复制整个数组。（基本和py一样啦...）
* **gpu加速：** `a=a.cuda()`，注意要对a进行赋值。
* **求最值：**
    * 对a的每一列求最值：使用`np.max(a,axis=0)`和`np.min(a,axis=0)`函数
    * 对a的每一行求最值：使用`np.max(a,axis=1)`和`np.min(a,axis=1)`函数
    * 返回值是一个1* n的array
    * 返回列表a中出现的所有不同元素:`np.unique(a,return_index=Falise,return_counts=False)`.当`return_index`为`true`时,同时返回每个unique元素出现的第一个index;`return_counts`为`true`时,同时返回每个unique元素出现的次数.
    * 从数组a中随机抽取元素(被抽样的数组必须一维的):使用`np.random.choice()`,例如`np.random.choice(a,size=(2,3),replace=True)`,size表述要抽取的元素的大小,replace为true表示允许重复抽取.

## 计算
### 范数计算
* 计算向量x和y的L2范数(欧式距离):`np.linalg.norm(x - y)`
### 统计学与概率论
* 对矩阵a进行均值计算:`np.mean(a,axis=0)`,axis=0时表示对列求均值,axis=1时表示对行求均值.
