# #4 支持向量机（SVM）
>——有人觉得这是现成的最好的分类器
***
![SVM示意图](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1527170673&di=1dcc82cd28132c6c53838482d49ce094&imgtype=jpg&er=1&src=http%3A%2F%2Fs1.sinaimg.cn%2Fmw690%2F002xlA0Pgy6TFXnKcVi70%26amp%3B690)
***
## 线性可分问题
要了解SVM的概念，让我们先从**线性可分**的问题谈起。什么是线性可分问题呢？以二分类问题为例，当一个分类器对数据集$\Bbb D=(x_1,y_1),(x_2,y_2),...(x_n,y_n)$进行分类时，假设其中任意样本$(x_i,y_i)$的特征向量$x_i$都是$m$维向量，那么数据集$\Bbb D$的特征就可以表示为$m$维向量空间中的一组点集。进行分类的过程其实就是找到一个**超平面**，可以把表示两种不同类别数据的点划分成相应的两部分$\{C_{+1},C_{-1}\}$。假如我们找到的超平面是线性的（例如二维空间的直线，三维空间的平面），那么这个分类问题就是线性可分问题
![](https://raw.githubusercontent.com/creeper121386/blog-file/master/%E6%B7%B1%E5%BA%A6%E6%88%AA%E5%9B%BE_%E9%80%89%E6%8B%A9%E5%8C%BA%E5%9F%9F_20180519150228.png)

事实上，对一个线性可分的数据集而言，这样的线性超平面不止一个。以上图中的数据为例，即使图中的直线略微左右倾斜，仍然可以正确地划分数据集。那么，在这些不同的直线中，我们需要找到一条最优的直线$L^*$。

所谓“最优”，指的是如果我我们向数据集中增添新的数据，直线仍然能很好的划分出两个类别，这就要求$L^*$正好处在两个类别的数据的“正中间”，换句话说，就是要求$\{C_{+1},C_{-1}\}$中靠近边界的点（这些点称作**支持向量点**）的距离$L^*$最近，也即求解最大间隔超平面。这就将求解$L^*$的问题转化为一个极值问题。

## 求解最大分隔超平面

### 函数间隔与几何间隔

假设我们要求的超平面为$L:f(x)=w^Tx+b$，其中$b$是截距，$w$是参数向量，且$w$的方向是超平面的法矢量方向，$x$表示点坐标（也即是特征值向量）。那么$\Bbb D$中的任一点$x_i$到$L$的距离为:
$$d_i=\frac{|w^Tx_i+b|}{||w||}$$

由于是二分类问题，我们使用$+1$和$-1$来标记样本的正反类。如果分类器能够正确分类的话，对样本$(x_i,y_i)$，有:
* 对于$y_i=+1$，有${w^Tx_i+b}>0$
* 对于$y_i=-1$，有${w^Tx_i+b}<0$

也就是说，无论$y$的取值如何，只要分类器能够正确分类，就有：
$$y_if(x_i)=y_i(w^Tx_i+b)\geqslant0 \qquad(1)$$

令$\gamma_i=y_if(x_i)$，$\gamma_i$称为点$x_i$到$L$的函数间隔。由于$|y|=1$，可以推出$\gamma_i=|w^Tx_i+b|$，进而得出：
$$d_i=\frac{\gamma_i}{||w||}$$

（这也是为什么要把$y$设定为$1$和$-1$，而不是像逻辑回归中一样设置为$0$和$1$的原因。）
这样就完成了求距离的工作，接下来考虑求解超平面$L$的最优解$L^*$。

### 拉格朗日算子法
还记得我们的目的吗？我们要求的是到支持向量点的距离最近的超平面$L^*$。首先我们要表示出支持向量到$L$的距离。假设支持向量到$L$的函数间隔为$\gamma_v$，那么要求解的极值问题就是：
$$\max_{w,b} \frac{\gamma_v}{||w||},\qquad(2.1)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant\gamma_v,\; (x_i,y_i)\in\Bbb D \qquad(3.1)$$

其中式$(2)$是目标函数，式$(3)$是限制条件。式$(2)$的由来是：由于$\gamma_v$是支持向量到$L$的函数间隔，应当是所有点到$L$的函数间隔中最小的，因此任一点$x_i$到$L$的函数间隔$y_i(w^Tx_i+b)\geqslant\gamma_v$。由于$\gamma_v\geqslant0$，因此同时也保证了条件$(1)$，也即保证了我们求得的$L^*$是一个正确的分类器。

接下来对该极值问题进行简化：
* $w$是最终要求的变量，求$\max \limits_{w,b}\frac{\gamma_v}{||w||}$等价于求$\min \limits_{w,b}\frac{1}{2}{\gamma_v}||w||^2$
* 由于$\gamma_v$的取值并不影响最终求到的$L^*$，因此令$\gamma_v=1$。

得到简化后的极值问题：
$$\min_{w,b}\frac{1}{2}||w||^2,\qquad(2.2)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant1,\; (x_i,y_i)\in\Bbb D \qquad(3.2)$$

这是一个凸二次优化问题。为了求解这类极值问题，我们可以使用**拉格朗日算子法**。构造拉格朗日函数：
$$\mathcal L(w,b,\alpha)=\frac12||w||^2+\sum_{i=1}^n \alpha_i(1-y_i(w^Tx_i+b))$$

其中$[\alpha_1,\alpha_2,...\alpha_n]$是拉格朗日算子。根据拉格朗日算子法，有以下等式关系：
$$
\begin{cases}
\frac{\partial \mathcal L}{\partial w}=0\\[2ex]
\frac{\partial \mathcal L}{\partial b}=0
\end{cases}
\;\Rightarrow\;
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i\\[2ex]
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

可以看到，当$\alpha_i>0$时，必有$\gamma_i=1$，也即对应的$x_i$必然是支持向量。事实上，$L^*$的取值也仅与支持向量有关，其他数据的分布并不影响超平面的选取（这也是SVM的好处之一）。到这里我们就初步得到了求解$L^*$所需要的条件，包括式$(2.2),(3.2),(4),(5.2)$。接下来只要根据这些条件求解得到$w$和$b$，就可以得到$L^*$的方程。稍后，我们将对这个问题进行进一步优化，然后给出求解的详细步骤。

### 软间隔与松弛变量

注意到，上述数学模型假设最优超平面$L^*$必定可以准确无误地把$\{C_{+1},C_{-1}\}$分隔开，但事实上，我们使用的训练数据中往往会有一些反常样例，这种情况下要求$L^*$把所有的数据都准确地分出来是不现实的，反而可能会导致过拟合问题。因此，我们可以允许少数样本点落在$L^*$不属于它所在类别的一侧。

为此，引入**软间隔**的概念。所谓软间隔，也就是不太严格的分类器，它允许少数样本不满足约束条件$(3.2):y_i(w^Tx_i+b)\geqslant1$。为了达到这一目的，对每一个样本点$(x_i,y_i)$都引入松弛变量$\xi_i \geqslant0$，使得对$(x_i,y_i)$的约束条件变为$y_i(w^Tx_i+b)\geqslant1-\xi_i$，此时$\xi_i$可以反映样本$(x_i,y_i)$允许偏离$L^*$的程度。这样对于那些靠近$L^*$的点，分类器就有了容错的空间。

当然，这些不满足约束条件的点也应当尽可能少。因此，我们在最小化目标函数$\frac{1}{2}||w||^2$时，也应当同时最小化$\xi$的值。因此，在目标函数中添加一项来描述各个样本的$\xi$的总和，目标函数和约束条件就变为：
$$\min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i,\qquad(2.3)$$

$$ s.t.\;\ y_i(w^Tx_i+b)\geqslant1-\xi_i,\; (x_i,y_i)\in\Bbb D,\xi_i\geqslant0 \qquad(3.3)$$

其中$C$是一个事先确定好的常数，用于控制目标函数中两项(“寻找间隔最大超平面”和“保证数据点偏差量最小”)之间的权重。由于目标函数和约束条件发生了变化，因此构造新的拉格朗日函数：
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
\alpha_i\geqslant0,\;\mu_i\geqslant0,\\
y_if(x_i)-1+\xi_i\geqslant0,\\
\alpha_i(y_if(x_i)-1+\xi_i)=0,\\
\xi_i\geqslant0,\; \mu_i\xi_i=0.
\end{cases}
\qquad(5.3)
$$

根据式$(5.3)$可以得到，当$\alpha_i$的取值不同时，要满足的式子也不同。与之前类似，不同$\alpha$的取值下，要满足的条件分别为：
$$
\begin{cases}
y_if(x_i)\geqslant1,\qquad\text{if }\;\alpha_i=0  \\
y_if(x_i)=1,\qquad\text{if }\;0<\alpha_i<C \\
y_if(x_i)\leqslant1,\qquad\text{if }\;\alpha_i>C 
\end{cases}
\qquad(5.4)
$$


## 核函数

### 映射到高维空间
还记得我们的出发点是线性可分问题吗？事实上，之前的内容都是针对线性可分问题而言的。但是在实际的数据处理中，数据分布往往不是线性可分的。这种情况下我们无法通过一个线性超平面来分隔数据。

对于线性不可分问题，不同的机器学习算法有不同的处理方法。而SVM的处理方法是：使用一个映射函数$\phi$，把原数据映射到某一个更高维的特征空间，使得在这个空间中，数据变得线性可分。（事实上，必定存在某个维度，使得映射后的数据线性可分。）以下图为例，一组线性不可分的二维数据经过$\phi$映射到三维空间，在新的特征空间中，数据变得线性可分。之后，只要在新的特征空间中列出相应的拉格朗日方程，求解极值问题即可得到$L^*$。
![](https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=3799746487,1412905946&fm=27&gp=0.jpg)

我们之前在极值问题中得到过消去$w$，使用$\alpha$表达的$L$的方程：$f(x)=\sum_{i=1}^n\alpha_i y_i\left \langle x_i,x\right\rangle+b$。在数据映射到高维空间后，方程变为：
$$f(x)=\sum_{i=1}^n\alpha_i y_i\left \langle \phi x_i,\phi x\right\rangle+b\qquad(6)$$

其中$\phi x_i,\phi x$是映射后的特征向量和自变量。可以看到，其中涉及到了$\phi x_i$和$\phi x$内积的计算，也即是矩阵的乘法计算。这就面临一个问题：如果映射之后的特征空间维度很高，那么进行矩阵计算将会相当耗时。为了解决这一问题，我们使用核函数的方法。

### 核方法

核方法的思想是：为了避免在高维空间中进行矩阵运算，那么在原空间中是否可以找到这么一个函数$k(x_i,x)$，使得$k(x_i,x)$恰好等于高维空间中$\left \langle x_i,x\right\rangle$的值呢？如果函数$k(\dot\;,\dot\;)$存在的话，就可以直接在低维空间中计算内积，不必计算高维向量的乘法了。这个函数$k$就称为**核函数**。

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

可以证明，只要对于任意数据集$D$，$K$都是半正定矩阵，那么$k(\dot\;,\dot\;)$就可以作为核函数。此外，两个核函数的线性组合、两个核函数的直积也是核函数。下面给出一些常用的核函数：
|函数名称|表达式|参数|
|:--:|:--:|:--:|
|线性核|$k(x,y)=x^Ty$|$None$|
|多项式核|$k(x,y)=(x^Ty)^d$|$d\geqslant1$，是多项式的次数|
|高斯核|$k(x,y)=exp(-\frac{\|x-y\|^2}{2\sigma^2})$|$\sigma>0$，是高斯核的带宽|
|拉普拉斯核|$k(x,y)=exp(-\frac{\|x-y\|}{\sigma})$|$\sigma>0$|
|Sigmoid核|$k(x,y)=tanh(\beta x^Ty+\theta)$|$tanh$是s双曲正切函数，$\beta>0,\theta<0$|

上述函数都满足核函数的条件，也即$k(x_i,x)=\left \langle x_i,x\right\rangle$。使用核函数的好处在于，当我们使用特定的核函数$k$时，就已经隐式地把数据映射到了某一个高维空间中去了，不必再进行复杂的映射变换，也无需关注映射后的数据是怎样分布的。使用不同的核函数，就可以把数据映射到不同的高维特征空间。因此，核函数的选取就成为了影响SVM性能的一个重要因素。

进行核函数映射之后，极值问题中所有涉及到向量内积的计算，就可以全部用核函数代替，接下来，就可以进行对极值问题的具体求解了。

## 序列最小化优化算法(SMO)

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

这就是我们所拥有的所有条件。将式$(4.2)$代入$(2.3)$，可以消去$w,b$,得到该极值问题的对偶问题：
$$\max_{\alpha}\sum_{i=1}^n\alpha_i-\frac12 \sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jk(x_i,x_j),\qquad(2.4)$$

$$ s.t.\sum_{i=1}^n\alpha_iy_i=0,\; (x_i,y_i)\in\Bbb D,\;0\leqslant\alpha\leqslant C \qquad(3.4)$$

该问题和原来的极值问题完全等价。可以看到，对偶问题中，只有向量$\alpha$一个变量，只要求出$\alpha$，就可以根据
$$
\begin{cases}
w=\sum_{i=1}^n\alpha_ix_iy_i\\
b=-\frac12{(\max_{i:y_i=-1}w^Tx_i+min_{i:y_i=1}w^Tx_i)}
\end{cases}
\qquad(7)
$$

来得到$w$和$b$，进而得到最优超平面$L^*$。

### 求解$\alpha$


## 支持向量回归


    