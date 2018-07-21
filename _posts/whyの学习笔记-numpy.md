---
title: whyの学习笔记:各种常用的Python库（已更新Numpy, Re, matplotlib）
date: 2018-04-11 19:10:51
tags:
- 编程
- python
categories: python

---

# whyのnmupy学习记录！

#### 导入：
```python
import numpy as np
```
## 数据类型
**数组（array）：** 建立数组：np.array(列表)，将列表变成array。

***

<!-- more -->

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

* 删除矩阵a的特定行行/列:
    * 删除行: 　`np.delete(a,index,axis=0)`
    * 删除列:　`np.delete(a,index,axis=１)`
    * index表示要删除的行/列的索引,可以为数字(删除一行/列),也可以为list(删除多个指定行/列)

* 获取矩阵a的行列数:`np.shape(a)`,返回值是一个tuple.
* 获取数组a中沿着轴x方向,最大值/最小值的索引（如果是一维数组,则不必考虑方向问题）：

|     |沿行方向最值         |沿列方向最值 |
|:---:|:-----------------:|:-----------------:|
|最大值|np.argmax(a,axis=1)|np.argmax(a,axis=0)|
|最小值|np.argmin(a,axis=1)|np.argmin(a,axis=0)|


## 计算
### 范数计算
* 计算向量x和y的L2范数(欧式距离):`np.linalg.norm(x - y)`
### 统计&概率
* 对矩阵a进行均值计算:`np.mean(a,axis=0)`,axis=0时表示对列求均值,axis=1时表示对行求均值.
## 文件I/O
* `np.loadtxt('文件名',dtype='',delimiter='分割符')`,从指定的文本文件中读取数据,返回值是array类型.可以指定分隔符读取,也可以读取csv文件.
* 保存&加载`array`:
    * `np.savetxt('path', array)`
    * `np.save('path', array)`，以二进制形式保存。`path`中的文件名后缀应该是`.npy`。相应的加载方法是`array = np.load('path')`
    * `np.savez('path', name1 = array1, name2 = array2...)`用来保存多个数组。文件后缀应当是`.npz`。也用`a = np.load(path)`加载，但返回的`a`是一个字典。使用`a['name1']`来获取相应的数组。

***

# whyの正则表达式学习记录！

![不说废话直接上图](https://images.cnblogs.com/cnblogs_com/huxi/Windows-Live-Writer/Python_10A67/pyre_ebb9ce1c-e5e8-4219-a8ae-7ee620d5f9f1.png)

>导入：`import re`

## pattern类

`pattern类`指的是用于匹配的模式。

* 将字符串形式的正则表达式编译为`pattern类`：`p = re.compile('正则表达式',匹配模式)`，返回的`p`是一个`pattern类`。
* 使用已有的模式`p`进行匹配：`match = p.match('要匹配的字符串',匹配模式)`，返回值`match`是匹配的结果（是一个`SRE_Match`对象），如果没有匹配项，则返回`NoneType`。
* 对匹配结果`match`（`SRE_Match`对象），使用`match.group()`来查看成功匹配的字符串。
* 正则表达式也可以不编译直接匹配，使用`re.match(正则表达式,待匹配字符串,匹配模式)`函数。

#### 注：
* 上述`匹配模式`指的是忽略大小写、匹配多行之类的选项：
    * re.I 忽略大小写
    * re.M 多行模式
    * re.S 即为`.`并且包括换行符在内的任意字符（`.`不包括换行符）
    * re.L 表示特殊字符集 `\w, \W, \b, \B, \s, \S` 依赖于当前环境
    * re.U 表示特殊字符集 `\w, \W, \b, \B, \d, \D, \s, \S` 依赖于Unicode 字符属性数据库
    * re.X 为了增加可读性，忽略空格和`#`后面的注释
* 在`python`中使用正则表达式时，由于`python`中的字符串本身也用`\`进行转义，因此在字符串形式的正则表达式前一般加上前缀`r`，表示该字符串不转义。

## API

* 使用正则表达式分割字符串：`re.split(用来匹配分割部分的正则表达式,待分割字符串)`
* `re.findall(正则表达式,待匹配字符串)`以列表形式返回所有匹配的子串。 

***

# 使用Matplotlib数据可视化！

## 二维函数图像

* 导入：`import matplotlib.pyplot as plt`
* 建立一个图像**窗口**`plt.figure(num, figsize=(a, b))`
    * 其中`num`表示编号，`figsize`表示画布大小。
    * `figure`是一次性的，在`plt.show()`之后就会销毁。
* 画二维函数线图：`plt.plot(x, y, color='red', linewidth=w, label='name', linestyle='--')`，其中`x, y`都是可迭代对象（如列表，数组等）。
    * 为了画出函数图形，一般使用`x = np.linspace(a, b, n)`来产生区间$(a,b)$之间`n`个均匀的值，然后将y表示成x的函数关系。
    * `color='red'`表示图像的颜色。
    * `linestyle='--'`表示线的类型，`'--'`表示虚线。
    * `linewidth`表示线宽
    * `label`表示线的名称（标签），做图例时会用到。
* **添加图例**：plt.legend(loc='upper right')
    * `loc`表示图例位置。`loc='best'`可以自动放到最佳位置。

### 调整坐标轴

* `plt.xlim((a, b))`表示x轴显示范围是$(a,b)$，使用`plt.xlabe('name')`调整x轴名称。y轴同理。
* 调整刻度&给刻度添加名称：`plt.xticks(ticks, names)`，ticks是刻度值组成的列表，`names`是相应名称组成的列表
* `axis = plt.gca()`获得坐标轴当前的状态，`axis`的成员`xaxis`和`yaxis`表示两个坐标轴，`apines`表示边框。`axis`有以下方法：
    * `axis.spines['top'].set_color('red')`设置上边框颜色，下、左、右同理。
    * `axis.spines['top'].set_position(('data', 0))`设置上边框的位置,`(data,0)`表示放在`y=0`处。
    * `axis.xaxis.set_ticks_position('top')`设置坐标刻度名称的位置，可以有`top，bottom，both，default，none`
* 例如，把`x，y`轴都调整到原点处：
    ```python
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ```
* 使用`plt.xticks(())`隐藏x轴，y轴同理。

### 添加标注

* `plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)` 在某一点$(x_0,y_0)$处画出一条垂直于x轴的虚线.
* 为某一点设置样式`plt.scatter([x0, ], [y0, ], s=50, color='b')`，其实`plt.scatter`是画散点图用的...

### 散点图

* `scatter(x, y, s=75 c=None, marker=None, cmap=plt.cm.hot, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None)`
    * `x,y`分别表示各个点的横纵坐标，都是可迭代对象
    * `s`表示`size`，点的大小，int
    * `c`表示颜色，颜色可以是一个长度等于散点数的列表，每个元素用来描述每个点的颜色。
    * `cmap`表示颜色组，设置颜色组可以自动配色。`cmap=plt.cm.hot`表示暖色组。所有的`cmap`在[这里](https://matplotlib.org/examples/color/colormaps_reference.html)
    * `aplpha`表示透明度
    * `marker`表示形状
    * `edgecolors`表示描边颜色

### 条形图

* `plt.bar(x, y, edgecolor, facecolor)`
    * `x`是各个条形的横轴值，一般是`np.arange(10)`
    * `y`时条形的柱高，正在上，负在下。
    * `edgecolor`表示描边颜色， `facecolor`表示柱体颜色。
    * 部分参数（如透明度之类的）和散点图一致。
    * 条形图不能使用颜色组。

### 等高线图

* `plt.contourf(X, Y, Z, cmap, alpha)`将三维空间的点画成等高线图。
* 这次的`X,Y,Z`不是普通的数组。生成方式如下：

    ```python
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X,Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ```
    * 等高图可以使用颜色组`cmap`
    * `aplpha`表示透明度
* 使用`C = plt.coutour(X, Y, Z, 8, colors='black', linewidth=0.5)`进行描边。其中数字8表示等高线密度。
* 使用`plt.clabel(C, inline=True, fontsize=10)`为描边添加高度数字。
    * `C`是之前`coutour`返回的对象。
    * `inline`表示是否在线上，`fontsize`是字体大小。 

## 显示图片

* `plt.imshow(img)`来把图片加入画布，img可以是`numpy`数组
* 使用`plt.colorbar()`来添加一个颜色指示条。
* 记得隐藏坐标轴

## 3D图形

* 导入额外模块`from mpl_toolkits.mplot3d import Axes3D`
* 定义一个画布`fig = plt.figure()`
* 添加3D坐标轴：`ax = Axes3D(fig)`
* 准备`x,y`数据（任意方法），得到等长的一维数组`x,y`
* 把`x，y`编制成格栅`X, Y = np.meshgrid(x, y)`
* 获取`Z`值，注意`Z`必须从`X，Y`得到，而不是`x，y`
* 使用`ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), edgecolors)`来绘制曲面
    * `restride`和`cstride`是颜色网格的大小
    * `cmap`是颜色组
    * `edgecolors`描边颜色

## 子图

* `plt.subplot(a,b,c)`表示将整个`figure`分成`a`行`b`列，当前正在第`c`个子图上。
* 在子图上的绘制过程和之前相同。
* 可以使用`plt.savefig('path')`来保存图像。