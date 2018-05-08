# #3 多层全连接网络与BP算法

>科学是通过一次又一次的葬礼前进的　　　　　——普朗克
***
![神经网络示意图](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1525757979450&di=758e34fbd42cc487788231e77ca5eae5&imgtype=0&src=http%3A%2F%2Fstatic.open-open.com%2Flib%2FuploadImg%2F20170907%2F20170907172602_811.png)

## 神经元模型

当我们研究机器学习时，我们希望机器可以变得更加智能——自然的想法是通过模拟人脑的真实结构来达成这一点。但是要知道，人脑的结构非常非常复杂，直接模拟大脑似乎是一件不可能的事。所以让我们从最简单的单位——神经元开始吧。与生物体中的神经元类似，神经元模型之间会互相连接并传递信号，神经元可以认为是一个完成 **“接受信号->对接受的信号进行处理->把处理后的信号发出”** 的工作单元。这样看来，一个神经元的本质就是一个函数，其输入参数为传入的信号，输出值为传出的信号。这样就可以得到简单的神经元模型：

![神经元模型示意图](https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=2956569879,4024414811&fm=27&gp=0.jpg "神经元模型示意图")

这就是经典的**M-P神经元**。我们记图中表示的神经元为$c$，可以看到$c$接受$n$个输入值，将输入值经过特定的计算后，发出一个输出值。其中要注意：

* 神经元之间的连接是加权的——也就是说，上一个神经元的输出值$x_i$在传递给$c$之后，需要先乘以一个对应**权重值**$w_i$，这用来描述不同输入值的重要程度。
* 神经元本身有一个**偏置值**，在图中表示为$b$，用来描述神经元被激活的难易度。假设$c$的激活阈值是$\theta$，那么有$-b=\theta$.也就是说，神经元接受的输入超过$-b$时就被激活。

* 神经元$c$将接受到输入值累加起来作为总输入值，并与阈值$-b$进行比较，还要将得到的值放入一个激活函数$f$中，再输出最终的输出值。$f$的选取有很多种，其中比较常用的有sigmoid函数：

$$\sigma(x)=\frac{1}{1-e^{-x}}$$

之前提到过，一个神经元的本质就是一个函数。于是，可以得出M-P神经元所表示的函数就是：
$$c(x_1,x_2...x_n)=f\left(\sum_{i=1}^{n}w_ix_i+b \right)\qquad(1)$$

从式$(1)$可以看到，M-P神经元表示的函数中，直到传值给激活函数之前，进行的都是简单的线性运算。这样除了计算方便以外，还有一个最大的好处：可以使用矩和向量来进行计算。

## 多层全连接神经网络

有了神经元模型，我们就可以在此基础上构建神经网络了。虽然叫做神经网络，但事实上它和真正的神经系统仍有很大差别。因为是由神经元构成的网络，自然就叫做“神经网络”了。

神经网络的种类极其丰富，不同的神经网络可以用来完成各种各样的机器学习任务。在这里，我们使用最简单的神经网络：多层全连接网络。多层全连接网络的结构像下图那样，由若干层组成(假设共有$n$层)，每一层又有若干个神经元，神经元之间彼此连接，其中：

* 相邻两层之间的任意两个神经元均有连接（全连接）
* 同一层之间的神经元没有连接
* 不相邻层之间的神经元没有连接
* 最左边的一层是输入层，最右边的一层是输出层，其他各层称为隐层。

![神经网络示意图](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1525781191880&di=35e9bc324d05d2f747030190044466f4&imgtype=0&src=http%3A%2F%2Fwww.huanqiujiemi.com%2Fcontent%2Fuploadfile%2F201609%2F4940954822171358351.jpg)

我们知道，每一个神经元都有若干个输入值和一个输出值。对第$i$层中第$j$个神经元$c_{ij}$来说：

* 它的输入值是$i-1$层的所有神经元的输出值（因为$c_{ij}$与上一层的神经元全连接）
* 输出值作为$i+1$层中所有神经元的输入值的一部分。

于是，以数据分类任务为例，我们将一个样本的各项特征值作为输入层各个神经元的输入值（也就是整个神经网络的输入值。这也要求输入层神经元个数要与样本的特征数相等），样本数据从第一层开始，沿着网络中的神经元不断传递，最终在输出层输出的值就是最终的预测结果（这就要求输出层的神经元个数要与分类的类别数相同），以此来完成预测任务。

如果我们从宏观上以向量和矩阵的思维来看这一步骤，又会有一些新的发现。假设第$i$层有$m_i$个神经元，第$i-1$层有$m_{i-1}$个神经元，那么：

* 将第$i$层各个神经元的输入值看作$m_{i-1}\times 1$的向量$x$。
* 每两层之间的神经元的连接是带有权重的，由于第$i$层的神经元和第$i-1$层的神经元之间全连接，在这两层之间就有$m_{i-1}\times m_{i}$个连接，也就有$m_{i-1}\times m_{i}$个对应的权重，我们把这些权重看作$m_{i-1}\times m_{i}$的权重矩阵$W$。
* 对第$i$层的$m_i$个神经元而言，共有$m_i$个偏置值，把这些偏置值看作$m_i\times 1$的偏置矩阵$b$

我们把神经网络的第$i$层看作函数$L_i$，根据神经元函数式$(1)$，可以得到：

$$L_i(x)=f\left(Wx^T+b^T\right)$$

因此，一个样本数据在神经网络中的传递，就是不断进行矩阵乘法和加法，以及调用激活函数计算的过程。这个过程称为**前向传播（forward）**。接下来的问题是：我们该如何得到适当的权重矩阵和偏置矩阵呢？在网络建立之初，网络的参数一般使用随机赋值的方法来初始化，之后再通过训练不断修正。这里我们使用python类来描述网络的结构，并写出前向传播的方法：

```python
class Net(object):
    def __init__(self, size):
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.random.randn(num) for num in size[1:]]
        self.num = len(size)
        self.size = size

    def pred(self, feat):
        tmp = feat
        nerve = [feat]
        for i in range(self.num-1):
            tmp = sig(np.dot(tmp, self.weights[i])+self.bias[i])
            nerve.append(tmp)
        return nerve
```

## 准备：数据导入及处理

有了网络结构，接下来进行数据的导入。将数据划分为训练集和测试集，导入成为numpy矩阵，并进行归一化处理。这里使用min-max归一化方法。进行归一化处理的代码如下所示：

```python
maxMat = np.row_stack((np.max(data, axis=0),)*num)
minMat = np.row_stack((np.min(data, axis=0),)*num)
data = (data-minMat)/(maxMat-minMat)
```

## 代价函数与梯度下降

### 代价函数

之前提到，神经网络的参数$\Theta$（包括所有的权重矩阵和偏置矩阵）的初始化是随机的。我们要想训练网络模型，就需要知道如何对参数进行调整。假设对于一个训练样本$(x,y)$（其中$x$是样本特征值，$y$是样本标签），我们希望通过调整参数$\Theta$，使得网络的预测结果能向着正确结果逼近。那么，假设有一个函数$E(y,y')$，可以描述当前的预测值$y'$和实际标签$y$之间的差距，我们只要向着使$E(x,y')$减小的方向调整参数就可以了。这个函数$E$就被称为代价函数。

在实际训练过程中，代价函数有很多选择，可以根据不同的任务来选取，在这里我们使用比较简单的均方误差来作为代价函数。假设网络的输出值$y'$是长度为$l$的向量，用$y_i'$表示$y'$的第i个元素，则：
$$E(y,y')=\frac{1}{2}\sum_{i=1}^l(y_i-y_i')^2$$

### 优化参数

代价函数表面看来只有$(y',y)$两个参数，但在计算$y'$的过程中，需要用到神经网络$N$的所有参数$\Theta$以及样本特征值向量$x$，也即$y'=N(\Theta,x)$。因此，$E(y,y')=E(\Theta,x,y)$。为了使得$E$的取值尽可能小，比较可行的方法是，向着“使得代价函数更小”的方向调整参数，就可以使神经网络的预测值向着正确值不断靠近。这里“使得代价函数更小的方向”用更加数学的语言来表示，也就是“代价函数的**负梯度**方向”。

我们知道，函数的梯度给出了函数增长最快的方向，因此它的反方向也就是下降最快的方向。向着负梯度方向调整参数，这就是传说中的梯度下降法。使用一个样本进行训练时，首先计算出代价函数，并求出代价函数对每一个参数（包括所有的权重和偏置值）的偏导数，所有参数的偏导项合起来就构成了梯度向量，梯度向量的每一项指示了每一个参数应该如何调节。

梯度下降法有很多不同的变种，比如动量法，adagrad等改进方法，此处我们以普通的梯度下降法为例，进行参数的优化。

## 误差逆传播算法(BP算法)

对于神经网络而言，梯度下降这个过程就变得有些复杂——因为神经网络有很多层，需要对大量的参数求偏导。而且从整体来看，不同层上不同参数的偏导数表达式也是不同的，如果直接盲目计算的话，将会耗费大量精力。因此引入了BP算法来解决求偏导这一问题。

### 计算图与链式法则

所有复杂的函数都可以分解为若干简单函数的组合，例如四则运算，指对函数，幂函数等等。我们可以用计算图模型将这些组合表示出来，图中的每一个结点表示一个简单函数。例如下图表示的复杂函数，可以用计算图简单地表示出来：
![一个简单的例子](https://raw.githubusercontent.com/creeper121386/blog-file/master/Screenshot-2018-5-8%20autoliuweijie%20DeepLearning.png)
（什么？你觉得还是原来更简单？计算机可不这么想...）


将函数分解为$+$或$*$这样的简单运算之后，求导和计算都变得很容易。计算图沿着箭头方向传播，是函数求值的过程；逆着箭头方向传播，就是函数求导的过程。这也是“逆传播”的含义。计算图的求导规则是：

* 对末端结点，可以根据函数值和该节点表示的简单函数直接求出偏导，并将偏导数传给下游结点。
* 中间节点接受从上游结点传来的偏导值，并乘以该结点对应的简单函数的偏导（本地偏导），将结果传递给下游节点。
* 初始结点接受上游传来的偏导值，并乘以本地偏导，得到最终关于某个参数的偏导。

可以发现，计算图运用的求导方法正是函数求导的链规则。通过将这一过程流程化，只要逆向遍历计算图，就可以得到对任何参数的偏导（无论该参数是在何处输入的），这也正是许多深度学习框架中所使用的方法。神经网络本身也是一个复杂函数。只要将它的计算图表示出来，就可以很容易地求出关于各个参数的偏导。

这样我们就解决了神经网络求导的问题。根据求导得到的梯度向量，就可以决定每个参数该增减多少，这样就实现了神经网络在一个样本下的训练过程。对一个训练样本进行参数优化的代码如下：

```python
def update(self, label, nerve, alpha):
        sigma = (label-nerve[-1])
        deltaW = [x*0 for x in self.weights]
        deltaB = [x*0 for x in self.bias]
        for i in range(self.num-1):
            grad = nerve[-1-i]*(-nerve[-1-i]+1)*sigma
            gradMat = np.row_stack((grad,)*len(nerve[-2-i]))
            Bn = np.column_stack((nerve[-2-i],)*len(nerve[-1-i]))
            deltaW[-1-i] += alpha*gradMat*Bn
            deltaB[-1-i] += -alpha*grad
            sigma = np.sum(grad*self.weights[-1-i])
        return deltaW, deltaB

```

接下来只要不断重复这个过程，每输入一个训练样本，神经网络就进行一次参数调整，如此循环，就可以完成网络的训练。整个BP算法训练过程的代码如下：

```python
def BP(self, trainData, trainLabel, epoch, batch, alpha):
        for j in range(epoch):
            for i in range(int(len(trainLabel))):
                deltaW = [x*0 for x in self.weights]
                deltaB = [x*0 for x in self.bias]
                batchData = trainData[i]
                batchLabel = trainLabel[i]
                nerve = [self.pred(x) for x in batchData]
                predLabel = [x[-1] for x in nerve]
                loss = np.sum((batchLabel-np.array(predLabel))**2)/2
                for x, y in zip(batchLabel, nerve):
                    w, b = self.update(x, y, alpha)
                    for k in range(self.num-1):
                        deltaW[k] += w[k]
                        deltaB[k] += b[k]
                for k in range(self.num-1):
                    self.weights[k] += deltaW[k]
                    self.bias[k] += deltaB[k]
            print('training: epoch', j, 'loss:', loss)
```

### 累积BP算法

按照BP算法，每输入一个样本，就要进行一次参数优化。但是这样的优化只针对当前的一个样本有效——也就是说，并不能提高网络在其他样本上的表现。因此，我们使用累积BP算法来解决这一问题。

所谓累积，指的是将整个训练集中所有样本的梯度向量累积起来，计算出每个样本上的梯度向量后，先不急着调整参数，而是等到把整个训练集中所有样本的梯度都计算过以后，取各个梯度向量的平均值，再根据平均值进行参数优化。这样就保证了每次优化都兼顾到了所有的数据，而不仅仅针对单独的训练数据。使用累积BP算法进行训练的代码如下所示：

```python
def AEBP(self, trainData, trainLabel, epoch, alpha):
        length = len(trainLabel)
        for j in range(epoch):
            deltaW = [x*0 for x in self.weights]
            deltaB = [x*0 for x in self.bias]
            nerve = [self.pred(x) for x in trainData]
            predLabel = [x[-1] for x in nerve]
            loss = np.sum((trainLabel-np.array(predLabel))**2)/length
            for x, y in zip(trainLabel, nerve):
                w, b = self.update(x, y, alpha)
                for k in range(self.num-1):
                    deltaW[k] += w[k]
                    deltaB[k] += b[k]
            for k in range(self.num-1):
                self.weights[k] += deltaW[k]/length
                self.bias[k] += deltaB[k]/length
            print('training: epoch', j, 'loss:', loss)
```

## 数据测试

数据测试所用到的前向传播方法在之前的BP算法中已经提到了。输入一个测试样本，在网络中进行前向传播，输出层的输出结果就是样本的预测结果。使用测试集测试并计算准确率的代码如下所示：

```python
def test(self, testData, testLabel):
        count = 0
        for x, y in zip(testData, testLabel):
            tmp = self.pred(x)[-1]
            Pred = np.argmax(tmp)
            if Pred == np.argmax(y):
                count += 1
        acc = count/len(testLabel)
        return acc
```

至此，我们就完成了一个多层全连接网络！
***
$\textit{2018.5.8}\quad by\ why.$