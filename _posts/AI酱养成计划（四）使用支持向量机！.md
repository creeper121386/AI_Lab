---
title: AI酱养成计划（四）使用支持向量机！
date: 2018-05-28 18:48:17
tags:
- 机器学习
- python
categories: 机器学习
---

# #4 支持向量机（SVM）

>——有人觉得这是现成的最好的分类器

***

![SVM示意图](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1527170673&di=1dcc82cd28132c6c53838482d49ce094&imgtype=jpg&er=1&src=http%3A%2F%2Fs1.sinaimg.cn%2Fmw690%2F002xlA0Pgy6TFXnKcVi70%26amp%3B690)

***

<!-- more -->

**$\mathrm{Warning:}$ 以下内容包含大量数学公式，有任何不懂之处，请反复推敲数学公式。**

## 线性可分问题

要了解SVM的概念，让我们先从 **线性可分** 的问题谈起。什么是线性可分问题呢？以二分类问题为例，当一个分类器对数据集$\Bbb D=(x_1,y_1),(x_2,y_2),...(x_n,y_n)$进行分类时，假设其中任意样本$(x_i,y_i)$的特征向量 $x_i$ 都是 $m$ 维向量，那么数据集 $\Bbb D$ 的特征就可以表示为 $m$ 维向量空间中的一组点集。进行分类的过程其实就是找到一个 *超平面** ，可以把表示两种不同类别数据的点划分成相应的两部分 $\{C_{+1},C_{-1}\}$ 。假如我们找到的超平面是线性的（例如二维空间的直线，三维空间的平面），那么这个分类问题就是线性可分问题
![线性分隔超平面](https://raw.githubusercontent.com/creeper121386/blog-file/master/%E6%B7%B1%E5%BA%A6%E6%88%AA%E5%9B%BE_%E9%80%89%E6%8B%A9%E5%8C%BA%E5%9F%9F_20180519150228.png)

事实上，对一个线性可分的数据集而言，这样的线性超平面不止一个。以上图中的数据为例，即使图中的直线略微左右倾斜，仍然可以正确地划分数据集。在这些不同的直线中，我们需要找的就是一条最优的直线$L'$。

所谓“最优”，指的是如果我我们向数据集中增添新的数据，直线仍然能很好的划分出两个类别，这就要求   $L' $ 正好处在两个类别的数据的“正中间”，换句话说，就是要求 $ \{ C_{+1},C_{-1}\} $ 中最靠近边界的那些点（这些点称作 **支持向量点** ）距离  $L'$ 最近，也即间隔最大。因此最优超平面，指的也就是 **最大间隔超平面** 。这就将求解  $L'$ 的问题转化为一个极值问题。 求解最大间隔超平面，就是支持向量机（SVM）要解决的核心问题。

## 求解最大分隔超平面

### 函数间隔与几何间隔

假设我们要求的超平面为$L:f(x)=w^Tx+b$（要注意这里的$x$可能是高维向量，这取决于$m$的值），其中$b$是截距，$w$是参数向量，且$w$的方向是超平面的法矢量方向，$x$表示点坐标（也即是特征值向量）。那么$\Bbb D$中的任一点$x_i$到$L$的距离为:
$$d_i=\frac{|w^Tx_i+b|}{||w||}$$

由于是二分类问题，我们使用$+1$和$-1$来标记样本的正反类。如果分类器能够正确分类的话，对样本$(x_i,y_i)$，有:

* 对于$y_i=+1$，有${w^Tx_i+b}>0$
* 对于$y_i=-1$，有${w^Tx_i+b}<0$

也就是说，无论$y$的取值如何，分类器能够正确分类的充要条件是：

$$y_if(x_i)=y_i(w^Tx_i+b)\geqslant0 \qquad(1)$$

令$\gamma_i=y_if(x_i)$，$\gamma_i$称为点$x_i$到$L$的函数间隔。由于$|y|=1$，因此$\gamma_i=|w^Tx_i+b|$，进而得出：

$$d_i=\frac{\gamma_i}{||w||}$$

（这也是为什么要把$y$设定为$1$和$-1$，而不是像逻辑回归中一样设置为$0$和$1$的原因。）
这样就完成了表示距离的工作，接下来考虑求解超平面$L$的最优解$L'$。

### 拉格朗日算子法
还记得我们的目的吗？我们要求的是到支持向量点的距离最近的超平面$L'$。首先我们要表示出支持向量到$L$的距离。假设支持向量到$L$的函数间隔为$\gamma_v$，那么要求解的极值问题就是：
$$\max_{w,b} \frac{\gamma_v}{||w||},\qquad(2.1)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant\gamma_v,\; (x_i,y_i)\in\Bbb D \qquad(3.1)$$

其中式$(2.1)$是目标函数，式$(3.1)$是限制条件。式$(2.1)$的由来是：由于$\gamma_v$是支持向量到$L$的函数间隔，应当是所有点到$L$的函数间隔中最小的，因此任一点$x_i$到$L$的函数间隔都应当满足：

$$y_i(w^Tx_i+b)\geqslant\gamma_v$$

由于$\gamma_v\geqslant0$，因此式$(3.1)$同时也隐含了条件$(1)$，也即保证了我们求得的$L'$是一个正确的分类器。

接下来对该极值问题进行简化：

* $w$是最终要求的变量，这里有$\max \limits_{w,b}\frac{\gamma_v}{||w||}\iff \min \limits_{w,b}\frac{1}{2}{\gamma_v}||w||^2$
* 由于$\gamma_v$的取值并不影响最终求到的 $L'$ ，因此令$\gamma_v=1$。
（这一条可以试着自己证明一下呦！~~才不是因为我懒哼~~）

得到简化后的极值问题：
$$\min_{w,b}\frac{1}{2}||w||^2,\qquad(2.2)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant1,\; (x_i,y_i)\in\Bbb D \qquad(3.2)$$

这是一个凸二次优化问题。为了求解这类极值问题，我们可以使用**拉格朗日算子法**。构造拉格朗日函数：
$$\mathcal L(w,b,\alpha)=\frac12||w||^2+\sum_{i=1}^n \alpha_i(1-y_i(w^Tx_i+b))$$

其中$[\alpha_1,\alpha_2,...\alpha_n]$是拉格朗日算子，这里我们令$\alpha=[\alpha_1,\alpha_2,...\alpha_n]$。拉格朗日算子法在这里有以下等式关系：
$$
\begin{cases}
\frac{\partial \mathcal L}{\partial w}=0 \\[2ex]
\frac{\partial \mathcal L}{\partial b}=0
\end{cases}
\;\Rightarrow\;
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i \\[2ex]
0=\sum_{i=1}^n\alpha_iy_i
\end{cases}
\qquad(4.1)
$$

根据式$(4.1)$，可以消去$w$得到$L:f(x)=\sum_{i=1}^n\alpha_i y_i\left \langle x_i,x\right\rangle+b$，其中$\left \langle x_i,x\right\rangle$表示向量$x$与$x_i$的内积。这个表达式与之前的$L$表达式完全等价，之后就不必再考虑$w$，只要求出最优解对应的$\alpha$，就可以根据$(4.1)$计算得到$w$。另一方面，注意到式$(3.2)$是不等式约束关系。在拉格朗日算子法的使用中，出现不等式约束关系时，要求必须要满足**KKT条件(Karush-Kuhn-Tucker)**。在这里，对应的KKT条件是：

$$
\begin{cases}
\alpha_i\geqslant0,\\[2ex]
y_if(x_i)-1\geqslant0,\qquad\qquad(5.1)\\[2ex]
\alpha_i(y_if(x_i)-1)=0.
\end{cases}
$$

可以看到，对于不同取值的$\alpha_i$，要满足的KKT条件也不同。因此对于不同取值的$\alpha_i$，根据式$(5.1)$可以得到：

$$
\begin{cases}
y_if(x_i)\geqslant1,\qquad\text{if }\;\alpha_i=0  \\
y_if(x_i)=1,\qquad\text{if }\;\alpha_i\gt0 
\end{cases}
\qquad(5.2)
$$

可以看到，当$\alpha_i>0$时，必有$\gamma_i=1$，也即对应的$x_i$必然是支持向量。事实上，$L'$的取值也仅与支持向量有关，其他数据的分布并不影响超平面的选取（这也是SVM的好处之一）。到这里我们就初步得到了求解$L'$所需要的条件，包括式$(2.2),(3.2),(4),(5.2)$。接下来只要根据这些条件求解得到$w$和$b$，就可以得到$L'$的方程。稍后，我们将对这个问题进行进一步优化，然后给出求解的详细步骤。

### 软间隔与松弛变量

注意到，上述数学模型假设最优超平面$L'$必定可以准确无误地把 $\{C_{+1},C_{-1}\}$ 分隔开，但事实上，我们使用的训练数据中往往会有一些反常样例，这种情况下要求$L'$把所有的数据都准确地分出来是不现实的，这么强求反而可能会导致过拟合之类的问题。因此，我们可以允许少数样本点落在$L'$不属于它所在类别的另一侧。

为此，引入**软间隔**（或者称为“松弛变量”）的概念。所谓软间隔，也就是不太严格的分类器，它允许少数样本不满足约束条件$(3.2):y_i(w^Tx_i+b)\geqslant1$。为了达到这一目的，对每一个样本点$(x_i,y_i)$都引入松弛变量$\xi_i \geqslant0$，使得对$(x_i,y_i)$的约束条件变为$y_i(w^Tx_i+b)\geqslant1-\xi_i$，此时$\xi_i$可以反映样本$(x_i,y_i)$允许偏离$L'$的程度。这样对于那些靠近$L'$的~~在危险的边缘试探~~的点，分类器就有了容错的空间。

当然，这些不满足约束条件的点也应当尽可能少。因此，我们在最小化目标函数$\frac{1}{2}||w||^2$时，也应当同时最小化$\xi$的值。为此，在目标函数中添加一项来描述各个样本的$\xi$的总和，目标函数和约束条件就变为：

$$\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i,\qquad(2.3)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant1-\xi_i,\; (x_i,y_i)\in\Bbb D,\xi_i\geqslant0 \qquad(3.3)$$

其中$C$是一个事先确定好的超参数，用于控制目标函数中的两项(“寻找间隔最大超平面”和“保证数据点偏差量最小”)之间的权重。由于目标函数和约束条件发生了变化，因此构造新的拉格朗日函数：
$$\mathcal L(w,b,\alpha,\mu)=\frac12||w||^2+\sum_{i=1}^n \alpha_i(1-y_i(w^Tx_i+b))-\sum_{i=1}^n\mu_i\xi_i$$

分别各个变量求偏导数，得到：

$$
\begin{cases}
\frac{\partial \mathcal L}{\partial w}=0\\[2ex]
\frac{\partial \mathcal L}{\partial b}=0\\[2ex]
\frac{\partial \mathcal L}{\partial \xi}=0
\end{cases}
\;\Rightarrow\;
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i\\[2ex]
0=\sum_{i=1}^n\alpha_iy_i\\[2ex]
C=\alpha_i+\mu_i
\end{cases}
\qquad(4.2)
$$

同样的，由于约束条件是不等式，因此要满足KKT条件。此处的KKT条件为：
$$
\begin{cases}
\alpha_i\geqslant0,\;\mu_i\geqslant0,\\[2ex]
y_if(x_i)-1+\xi_i\geqslant0,\\[2ex]
\alpha_i(y_if(x_i)-1+\xi_i)=0,\\[2ex]
\xi_i\geqslant0,\; \mu_i\xi_i=0.
\end{cases}
\qquad(5.3)
$$

根据式$(5.3)$可以发现，当$\alpha_i$的取值不同时，要满足的条件也不同。与之前类似，不同$\alpha$的取值下，要满足的条件分别为：

$$
\begin{cases}
y_if(x_i)\geqslant1,\qquad\text{if }\;\alpha_i=0  \\
y_if(x_i)=1,\qquad\text{if }\;0<\alpha_i<C \\
y_if(x_i)\leqslant1,\qquad\text{if }\;\alpha_i>C 
\end{cases}
\qquad(5.4)
$$

$\alpha$在不同条件下，相应的$x_i,y_i$要满足不同的关系。从这里也可以看到，$\alpha_i$和$(x_i,y_i)$是一一对应的。

## 核函数

### 映射到高维空间
还记得我们一直讨论的是线性可分问题吗？之前的内容都是针对线性可分问题而言的。但是在实际的数据处理中，数据分布往往不是线性可分的（例如典型的 **“异或问题”** ）。这种情况下我们无法通过一个线性超平面来分隔数据。

对于线性不可分问题，不同的机器学习算法有不同的处理方法。而SVM的处理方法是：使用一个映射函数$\phi$，把原数据映射到某一个更高维的特征空间，使得在这个空间中，数据变得线性可分。（事实上，必定存在某个维度，使得映射后的数据线性可分。）以下图为例，一组线性不可分的二维数据经过某种变换$\phi$映射到三维空间，在新的特征空间中，数据变得线性可分。之后，只要在新的特征空间中列出相应的拉格朗日方程，求解极值问题即可得到$L'$。
![使用核函数映射到高维空间](https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=3799746487,1412905946&fm=27&gp=0.jpg)

我们之前在极值问题中得到过消去$w$，使用$\alpha$表达的$L$的方程：$f(x)=\sum_{i=1}^n\alpha_i y_i\left \langle x_i,x\right\rangle+b$。在数据映射到高维空间后，方程变为：
$$f(x)=\sum_{i=1}^n\alpha_i y_i\left \langle \phi x_i,\phi x\right\rangle+b\qquad(6)$$

其中$\phi x_i,\phi x$是映射后的特征向量和自变量。可以看到，其中涉及到了$\phi x_i$和$\phi x$内积的计算，也即是矩阵的乘法计算。这就面临一个问题：如果映射之后的特征空间维度很高，那么进行矩阵计算将会相当耗时。为了解决这一问题，我们使用核函数的方法。

### 核方法

核方法的思想是：为了避免在高维空间中进行矩阵运算，那么在原空间中是否可以找到这么一个函数$k(x_i,x)$，使得$k(x_i,x)$恰好等于高维空间中$\left \langle x_i,x\right\rangle$的值呢？如果函数$k(\dot\;,\dot\;)$存在的话，就可以直接在低维空间中计算内积，不必计算高维向量的乘法了。这样的函数$k$就称为**核函数**。

$k$的确是存在的，而且还不止一种。假设有核函数$k(\dot\;,\dot\;)$，那么定义核矩阵$K$：
$$
K=
  \begin{pmatrix}
   k(x_1,x_1) &  \cdots & k(x_1,x_n)\\[2ex]
  \vdots & \ddots & \vdots \\
   k(x_j,x_1)  & \cdots & k(x_j,x_n)  \\[2ex]
   \vdots & \ddots & \vdots \\
   k(x_n,x_1)  & \cdots & k(x_n,x_n)  \\
  \end{pmatrix}
$$

可以证明，如果对于任意的数据集$D$而言，$K$都是半正定矩阵，那么$k(\dot\;,\dot\;)$就可以作为核函数。此外，两个核函数的线性组合、两个核函数的直积也是核函数。下面给出一些常用的核函数：

函数名称|表达式|参数
:--:|:--:|:--:
线性核|$k(x,y)=x^Ty$|$None$
多项式核|$k(x,y)=(x^Ty)^d$|$d\geqslant1$，是多项式的次数
高斯核|$k(x,y)=exp(-\frac{\|x-y\|^2}{2\sigma^2})$|$\sigma>0$，是高斯核的带宽
拉普拉斯核|$k(x,y)=exp(-\frac{\|x-y\|}{\sigma})$|$\sigma>0$
Sigmoid核 | $k(x,y)=tanh(\beta x^Ty+\theta)$ | $tanh:$双曲正切函数，$\beta>0,0>\theta$

上述函数都满足核函数的条件，也即$k(x_i,x)=\left \langle x_i,x\right\rangle$。使用核函数的好处在于，当我们在低维空间中使用特定的核函数$k$时，就已经隐式地把数据映射到了某一个高维空间中去了，不必再进行复杂的映射变换，也无需关注映射后的数据是怎样分布的。如果使用不同的核函数，就可以把数据映射到不同的高维特征空间。因此，核函数的选取也就成为了影响SVM性能的一个重要因素。

进行核函数映射之后，极值问题中所有涉及到向量内积的计算，就可以全部用核函数代替（换言之，只要用核函数代替内积，就相当于完成了高维映射），接下来，就可以进行对极值问题的具体求解了。

## 训练模型：序列最小化优化算法(SMO)

### 对偶问题

接下来，我们就要对这个复杂的极值问题就行求解了。首先再次梳理该极值问题的条件。我们要求的目标函数和约束条件分别为：
$$\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i,\qquad(2.3)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant1-\xi_i,\; (x_i,y_i)\in\Bbb D,\xi_i\geqslant0 \qquad(3.3)$$

由该问题构造拉格朗日函数并对各项求偏导数，以及求出相应的KKT条件，再进行相应化简和推导，这些条件综合起来，得到的是式$(4.2),(5.4)$：

$$
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i\\
0=\sum_{i=1}^n\alpha_iy_i\\
C=\alpha_i+\mu_i
\end{cases}(4.2)\quad\&\quad
\begin{cases}
y_if(x_i)\geqslant1,\quad\text{if }\;\alpha_i=0  \\
y_if(x_i)=1,\quad\text{if }\;0<\alpha_i<C \\
y_if(x_i)\leqslant1,\quad\text{if }\;\alpha_i>C 
\end{cases}
\quad(5.4)
$$

并且由于引入了核函数，消去$w$，得到用$\alpha$表示的超平面$L$表达式为：
$$f(x)=\sum_{i=1}^n\alpha_i y_ik(x_i,x)+b\qquad(6)$$

这就是我们得到的所有条件。将式$(4.2)$代入$(2.3)$，可以消去$w,b$,得到该极值问题的对偶问题：
$$\max_{\alpha}\sum_{i=1}^n\alpha_i-\frac12 \sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jk(x_i,x_j),\qquad(2.4)$$

$$ s.t.\sum_{i=1}^n\alpha_iy_i=0,\; (x_i,y_i)\in\Bbb D,\;0\leqslant\alpha\leqslant C \qquad(3.4)$$

该问题和原来的极值问题完全等价。同时，根据已有条件也可以得到$w$和$b$关于$\alpha$的表达式：
$$
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i \\[2ex]
b=\frac{1}{|\Bbb S|}\sum_{i=1}^{|\Bbb S|}\left( y_i-\sum_{i=1}^{|\Bbb S|}\alpha_iy_ik(x_i,x_j)\right)
\end{cases}
\qquad(7)
$$

其中，$\Bbb S$表示所有支持向量的集合。$b$表达式的由来是：对任意支持变量，都有$y_if(x_i)=1$成立，因此根据该式求出一个$b_i$，然后对所有支持变量求得的$b_i$求和并取平均值。可以看到，在对偶问题中，只有向量$\alpha$一个变量，因此只要求出$\alpha$，就可以根据式$(7)$得到$w$和$b$，进而求出最优超平面$L'$。

### 求解$\alpha$

接下来我们就要开始求解对偶问题$(2.4),(3.4)$了。事实上，这个问题是一个凸二次规划问题，有许多现成的算法库可以调用。但是，直接求解这个问题会十分复杂——我们使用的训练数据量有时候会很大。因此我们需要一种简便的优化算法来求$\alpha$的值，这个算法就是**序列最小化优化算法(SMO)**（其实梯度下降法也可以呦(。・・)ノ）。

SMO算法的思想是，不直接去求解这个二次规划问题，而是先将向量$\alpha$初始化，然后然后再通过不断迭代，使得目标函数的值不断增大，最终逼近最大值。每一次迭代中，都只把$\alpha$的其中两个元素$\alpha_i,\alpha_j$视作变量，其他的$\alpha$视作常数，只优化这两个$\alpha$。具体的迭代步骤是： 

* 首先初始化$\alpha=[\alpha_1,\alpha_2...\alpha_n]=[0,0,0...0]$
* 在$\alpha$的$n$个元素中一选择一个$\alpha_i$作为第一个迭代对象（要怎么选呢？稍后会谈到）
* 在剩下的$n-1$个元素中选择一个$\alpha_j$作为第二个迭代对象（要怎么选呢？稍后会谈到）
* 暂且不管其他$\alpha$，根据已知条件求解得到$\alpha_i,\alpha_j$的值
* 选取两个新的$\alpha$继续进行迭代，直到所有$\alpha$都更新完毕

SMO算法看起来似乎比直接求解优化问题要更麻烦——AI酱可不这么认为o(￣ヘ￣o＃)！事实上SMO算法得益于可以快速收敛的优点，成为了SVM中的经典算法。SMO算法每次只选择两个$\alpha$进行更新，因此将多变量的优化问题变成了单变量优化问题（没错，虽然有两个$\alpha$但它的确是个单变量问题！稍后你就会看到这两个$\alpha$之间是有等式关系的），这样极大降低了优化问题的计算难度。SMO算法只要迭代进行单变量二次优化问题的求解，直到满足停止条件，就可以得到所有的$\alpha$值。

接下来以其中一次迭代为例，给出求$\alpha_i$和$\alpha_j$的步骤：

### 启发式方法

首先你会好奇，为什么每次要选择两个变量$\alpha$进行优化呢？一个不阔以吗？是的当然不阔以ヽ（≧^≦）ノ！我们要求优化之后的$\alpha$都满足约束条件$(4.2),(5.4)$，其中有一条：
$$0=\sum_{i=1}^n\alpha_iy_i$$

如果每次都只针对一个$\alpha$进行优化的话，无论如何也无法保证每次选取的$\alpha$之间都满足这个条件。因此，每次选择两个$\alpha$（假设用$\alpha_1$表示第一个优化对象，$\alpha_2$表示第二个优化对象），只要保证$y_1\alpha_1+y_2\alpha=\varsigma$，其中$\varsigma$是一个常数，就可以使得$0=\sum_{i=1}^n\alpha_iy_i$成立。

那么，在每一次迭代之前，要选择哪两个变量作为$\alpha_1,\alpha_2$呢？我们当然可以随机地进行选择，但我们总希望先选出当前情况下使得优化之后**效果最好**的两个变量，为此，我们采用**启发式**的选取方法。

首先回忆一下 **支持向量**的概念。之前曾经提到，超平面$L'$的表达式事实上只和支持向量有关。因此，优化支持向量$x_i$对应的$\alpha_i$带来的收益更大；而如果优化其他点对应的$\alpha$，带来的收益就小很多。我们知道，对支持变量$x_i$，有$y_if(x_i)=1$，而根据式$(5.4)$，有：
$$y_if(x_i)=1,\quad\text{if }\;0<\alpha_i<C$$

也就是说，当$\alpha$的取值在$(0,C)$之间时，对应的数据点是支持向量。因此，对于$\alpha_1,\alpha_2$的选取，都应当优先在$(0,C)$中选取，如果找不到$(0,C)$中的$\alpha$，再从其他区间选择。

对于$\alpha_1$和$\alpha_2$的选择，也要使用不同的方法。接下来进行分别讨论：

#### $\alpha_1$的选择
我们为所有的$\alpha$都赋予了初值$0$，SMO算法的目的是通过优化$\alpha$使得最终所有的$\alpha$都满足约束条件，并且在满足约束条件的前提下，使得目标函数最大。因此，我们只需要选择**违背约束条件最严重**的变量$\alpha_i$作为$\alpha_1$，并使得它在优化之后满足约束条件，那么就可以认为，优化这个$\alpha_i$的效果最好，或者说...优化程度最高（这有些类似梯度下降的做法）。

$\alpha$要满足的条件之前已经给出了(式$(4.2,(5.4)$)，其中：

* $(4.2)$是优化过程中要用到的条件，以及$w$和$\alpha$的关系，在这里无法作为判断依据；
* $(5.4)$是KKT条件，是关于$\alpha$的不等式约束条件，可以作为判断依据。因此，我们要选取的就是**违反KKT条件最严重**的变量$\alpha$作为$\alpha_1$。

衡量$\alpha$违反KKT条件的程度也很简单，例如对于$\alpha_i\in(0,C)$，要求$y_if(x_i)=1$，因此只要使得$y_if(x_i)$和$1$之间的距离最大就可以啦。据此，得出$\alpha_1$的选取方法：

$$
\alpha_1=
\begin{cases}
\mathop{argmax}_{\alpha_i} \;1-y_if(x_i),\quad\text{if }\;\alpha_i=0  \\
\mathop{argmax}_{\alpha_i} \;|y_if(x_i)-1|,\quad\text{if }\;0<\alpha_i<C \\
\mathop{argmax}_{\alpha_i} \;y_if(x_i)-1,\quad\text{if }\;\alpha_i>C 
\end{cases}
\qquad(8)
$$

在这三种情况中，我们又比较优先选择情况2（因为恰好是支持向量对应的$\alpha$），这样，就可以选出每一轮迭代的$\alpha_1$。

### $\alpha_2$的选择

选取得到$\alpha_1$之后，根据$\alpha_1$来选取$\alpha_2$。因为要优化的$\alpha_1$已经固定，我们只要选择使得优化收益最大的$\alpha_2$（也即使得目标函数上升最快的一个$\alpha$）就可以了。此时，这个问题就变成了一个单变量优化问题。

稍后我们进行的具体优化步骤会保证我们向着使得目标函数增长最快的方向优化$\alpha_1$和$\alpha_2$，因此选取$\alpha_2$时不必担心优化后目标函数不增反降，只要使得优化之后$\alpha_1$和$\alpha_2$的改变**尽可能大**就可以了。因此引入一个判断标准$E$。对样本$(x_i,y_i)$:
$$E_i=\underbrace{\sum_{i=1}^n\alpha_i y_ik(x_i,x_i)+b}_{f(x_i)}-y_i$$

当$\alpha_j$对应的$|E_1-E_j|$最大时，就能使得优化之后的变量$\alpha$改变最大，此时$\alpha_j$就是我们要选择的$\alpha_2$。也即：
$$\alpha_2=argmax_{\alpha_j}\; |E_1-E_j|$$

这样就完成了待优化参数$\alpha$的选择。选择$\alpha$的python代码如下所示：
```python
def chooseAlpha(trainData, trainLabel, b, alpha, index):
    E = 0
    num = len(index)
    predLst = [pred(trainData[x], b, alpha, kernel,
                    trainData, trainLabel) for x in index]
    for x in range(num):
        ix = index[x]
        if alpha[ix] > 0 and alpha[ix] < C:
            preE = E
            E = max(E, np.abs(1-predLst[x]*trainLabel[ix]))
            if preE != E:
                i = x
    if E == 0:
        for x in range(num):
            ix = index[x]
            preE = E
            if alpha[ix] == 0:
                E = max(E, 1-predLst[x]*trainLabel[ix])
            elif alpha[ix] == C:
                E = max(E, predLst[x]*trainLabel[ix]-1)
            if preE != E:
                i = x
    if E == 0:
        i = 0
    E = [predLst[x]-trainLabel[index[x]] for x in range(num)]
    j = np.argmax(np.abs(np.array(E) - E[i]))
    i = index[i]
    j = index[j]
    return i, j
```

### 优化$\alpha_i$和$\alpha_j$

接下来就可以对$\alpha_i$和$\alpha_j$进行优化了。首先根据条件$(4.2)$，得到$\alpha_i,\alpha_j$之间的关系式：

$$y_1\alpha_1^{old}+y_2\alpha_2^{old}=y_1\alpha_1^{new}+y_2\alpha_2^{new}=\varsigma\qquad(9.1)$$

其中$\alpha_1^{old}$和$\alpha_2^{old}$已知，要求的是$\alpha_1^{new},\alpha_2^{new}$，我们就可以将$\alpha_1^{new}$用$\alpha_2^{new}$表示（当然反过来也是可以的），只要求出$\alpha_2^{new}$就解决了对$\alpha$的优化问题。由于$y$的值只能是$\pm 1$，所以根据$y_1,y_2$的正负情况，上述条件变成：

$$
\begin{cases}
\alpha_1^{new}=-\alpha_2^{new}+\alpha_1^{old}+\alpha_2^{old} ,\quad if\ y_1=y_2\\
\alpha_1^{new}=\alpha_2^{new}+\alpha_1^{old}-\alpha_2^{old}
,\quad if\ y_1\ne y_2
\end{cases}
\qquad(9.2)
$$

可以认为$\alpha_1^{new}$和$\alpha_2^{new}$是一次函数关系。对应上述两种情况，$\alpha_2^{new}$与$\alpha_1^{new}$之间的函数关系可以用图形表示为：

![](https://raw.githubusercontent.com/creeper121386/blog-file/master/1042406-20161128221540099-1580490663.png)

上图中的$\alpha$之所以落在$[0,C]$的盒子内，是因为我们之前选取的$\alpha_1,\alpha_2$都是优先从$[0,C]$区间内选取的。根据图像可以看出，由于$\alpha_1^{new},\alpha_2^{new}$之间的约束关系$(9.2)$，使得$\alpha_2^{new}$是有上下界的。假设$\alpha_2^{new}$的上下界分别为$H,L$，根据图像有：

$$
\begin{cases}
L=max(0,\alpha_2^{old}-\alpha_1^{old}),H=min(C,C+\alpha_2^{old}-\alpha_1^{old})\quad if\ y_1\ne y_2\\
L=max(0,\alpha_2^{old}+\alpha_1^{old}-C),H=min(C,\alpha_2^{old}+\alpha_1^{old})\quad if\ y_1=y_2\\
\end{cases}
$$

因此，稍后我们求出$\alpha_2^{new}$之后，还要对其进行 **剪切**：如果$\alpha_2^{new}$的值不在$[L,H]$之间，需要把多余部分剪掉，即：

$$
\alpha_2^{new}=
\begin{cases}
H\qquad if\ \alpha_2^{new}>H \\
\alpha_2^{new} \quad if\ \alpha_2^{new}\in [L,H]\\
L \qquad if\ \alpha_2^{new}<L
\end{cases}
$$

这样得到的就是最终的$\alpha_2^{new}$。

### 求解$\alpha_2^{new},\alpha_1^{new}$

此刻，我们已经将最初的优化问题变成了仅与$\alpha_2^{new}$有关的单变量优化问题。事实上，这个优化问题就是一个简单的二次函数求最值问题，只不过目标函数稍微复杂一些。经过漫长的推导（真的很漫长...），就可以得到最终的只含有$\alpha_2^{new}$的目标函数，接下来求出目标函数在$\alpha_2^{new}$处的梯度，然后找到梯度为$0$的点，就可以使得目标函数$(2.4)$最大，得到$\alpha_2^{new}$的值。最终导出的方程式是：

$$(k(x_1,x_1) +k(x_2,x_2)-2k(x_1,x_2))\alpha_2^{new} = y_2((k(x_1,x_1) +k(x_2,x_2)-2k(x_1,x_2))\alpha_2^{old}y_2 +y_2-y_1 +g(x_1) - g(x_2))$$

$$\;\;\;\qquad \qquad = (k(x_1,x_1) +k(x_2,x_2)-2k(x_1,x_2)) \alpha_2^{old} + y2(E_1-E_2)$$

要注意，其中$x_1,x_2$表示的是$\alpha_1,\alpha_2$对应的$x$，$E_1,E_2$与之前提到的$E_i$计算方法一致。由上式可以得出更新$\alpha_2^{new}$的表达式：

$$\alpha_2^{new,unc} = \alpha_2^{old} + \frac{y2(E_1-E_2)}{k(x_1,x_1) +k(x_2,x_2)-2k(x_1,x_2))}$$

得到$\alpha_2^{new}$，就可以根据式$(9.2)$求出$\alpha_1^{new}$，这样，第一轮选择出来的$\alpha_1,\alpha_2$就优化完毕了。接下来只要循环这个过程，直到所有的$\alpha$都被更新。得到更新后的$\alpha$之后，只要根据式$(7)$，就可以得到$w,b$，进而得到最优超平面 $ L ^* $ ，这样就完成了SVM分类器。

接下来给出使用SMO算法进行训练的python代码（注意：以下代码封装的是一个连贯的函数，请连起来看0v0）。首先确定要使用的核函数，并将所有的$\alpha$值初始化：

```python
def train(trainData, trainLabel, num, kernel):
    # alpha = np.random.randint(0, C, (num, ))
    print('*** data training start ***')
    func = K[kernel] if kernel else np.dot
    alpha = np.zeros(num)
```
接下来开始循环选取适当的$\alpha_i,\alpha_j$进行优化，选取方法是直接调用之前的`chooseAlpha`函数。选取完毕后，首先计算出对应的$E_i,E_j$：
```python
    for t in range(loopTimes):
        index = [x for x in range(num)]
        b = 0
        while len(index):
            i, j = chooseAlpha(trainData, trainLabel, b, alpha, index)
            Ei = pred(trainData[i], b, alpha, kernel,
                      trainData, trainLabel)-trainLabel[i]
            Ej = pred(trainData[j], b, alpha, kernel,
                      trainData, trainLabel)-trainLabel[j]
```
接下来根据之前的内容，计算得出$\alpha_2^{new}$的临时解（未进行剪切）：
```python
            preI = alpha[i].copy()
            preJ = alpha[j].copy()
            yi = trainLabel[i]
            yj = trainLabel[j]
            xi = trainData[i]
            xj = trainData[j]
            if yi == yj:
                L = max(0, alpha[j]+alpha[i]-C)
                H = min(C, alpha[j]+alpha[i])
            else:
                L = max(0, alpha[j]-alpha[i])
                H = min(C, C+alpha[j]-alpha[i])
            # c = -sum([trainLabel[k]*alpha[k]
            #          for k in range(num) if k != i and k != j])
            eta = 2*func(xi, xj.T)-func(xi, xi.T)-func(xj, xj.T)
```
如果求得的上下界相等，或者得到的$\alpha_2^{new}$与原值几乎相等（无需更新），则直接跳过本轮循环。否则，对$\alpha_2^{new}$进行剪切，求出$\alpha_2^{new}$的解析解：
```python
            if L != H and eta < 0:
                alpha[j] -= yj * (Ei-Ej)/eta
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L
                if np.abs(alpha[j]-preJ) < 1e-3:
                    index.remove(i)
                    continue
```
接下来对$\alpha_1$进行更新，并根据式$(7)$计算$b$。如果当前更新的样本点是支持向量点，则求出$b$在本轮循环的平均值。最后从索引列表中删除当前$\alpha_i,\alpha_j$对应的索引，表示该索引对应的数据已经更新完毕。不断循环该过程，就可以完成训练，最终返回更新后的$\alpha,b$，训练结束。
```python
                alpha[i] += yi*yj*(preJ-alpha[j])
                b1 = b-Ei-yi*(alpha[i]-preI)*func(xi, xi.T) -\
                    yj*(alpha[j]-preJ)*func(xi, xj.T)
                b2 = b-Ej-yi*(alpha[i]-preI)*func(xi, xj.T) - \
                    yj*(alpha[j]-preJ)*func(xj, xj.T)
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                index.remove(i)
            else:
                index.remove(i)
                continue
        print('training: epoch No.', t)
    print('*** data training finish ***')
    return alpha, b
```

## 数据测试

完成分类器之后，数据测试就变得十分简单。在最开始，我们已经给出了SVM正确分类的充要条件，即条件$(1)$。因此只要将测试集数据输入SVM，判断式$(1)$是否成立即可。以下为数据测试的python代码：
```python
def test(testData, testLabel, trainData, trainLabel, alpha, b, kernel):
    print('testing...')
    count = 0
    for x, y in zip(testData, testLabel):
        if not kernel:
            w = cal_w(x, alpha, trainData, trainLabel)
            predLabel = np.dot(w, x.T)+b
        else:
            predLabel = pred(x, b, alpha, kernel, trainData, trainLabel)
        count += 1 if y*predLabel > 0 else 0
    return count/len(testLabel)
```
至此，我们终于完成了一个支持向量机！撒花 ★,°*:.☆(￣▽￣)/.*.°★* 

***

这篇博客从5.28开始，已经过去快半个月了...终终终终于写完啦！（泪流满面

$$2018.6.12\;by \; \mathcal{WHY}$$
