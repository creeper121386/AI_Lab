---
title: whyの学习笔记:Pytorch
date: 2018-06-10 20:47:46
tags:
- 深度学习
- pytorch
categories: 深度学习
---

## 基本类

### 张量（Tensor）:

一个tensor实例有以下常用成员：

* `Tensor.data`: 张量的数据本体
* `Tensor.grad`: 该张量处的梯度（如果有的话）
* `Tensor.grad_fn`: 指向在该张量处进行的函数计算（计算图中的计算节点），进行梯度反向传播时会用到。如果是由用户创建的tensor，则`grad_fn = None`。
* `Tensor.grad_fn.next_function`: 上一级节点处的`grad_fn`。

`tensor`有以下常用函数/方法：

#### 创建`Tensor`

* `torch.eye(n)`，返回一个单位阵。
* `torch.ones(input)`，返回一个维度为`input`的全`1`的`Tensor`
* `torch.zeros(input)`，返回一个维度为`input`的全`0`的`Tensor`
* `torch.t(input)`将`input`进行转置
* `troch.full(size, value, device)`返回

#### 常用操作

* 把`numpy`数组转换为`tensor`：`torch.from_numpy(array)`
* 将`Tensor`展开为特定的`size`：`tensor.view(x,y)`其中参数`x,y`是展开后的`size`。
    * 如果`x,y`中任意一者为`-1`，则表示该维度自动计算（例如`4x4`的张量`a.view(-1,8)`展开为`2x8`）
    * 如果`tensor.view(-1)`表示展开为`1*n`的张量。
* `tensor.squeeze(n)`表示若`tensor`的第`n`维度是`1`，则去掉该维度（例如`3x3x1`张量`a.sqeeze(3)`变成`3x3`）
* `tensor.unsqueeze(n)`：`sqeueze`的逆操作，参数含义相同。
* 取最大值：`torch.max(tensor, axis)`
    * `axis=1`时返回每一行的最大值以及对应的列索引
    * `axis=0`时返回每一列的最大值以及对应的行索引
    * 不传入`axis`，直接用`torch.max(tensor)`，返回所有元素中的最大值。


<!-- more -->

### Autograd

* 设置`Tensor.requires_grad = True`时，pytorch会自动追踪对该张量进行的计算。只要调用`Tensor.backward()`即可反向计算出所有节点处的梯度。
* 如果不想追踪对张量的计算， 可以使用：
    * `Tensor.detach()`，把该张量从计算图中分离。
    * 使用`with torch.no_grad():`来包装代码。使用示例：
        ```python
        with torch.no_grad():
            [我是代码]
        ```

使用反向传播`Tensor.backward()`：

* `Tensor.backward(gradient=grad)`： grad是一个张量，用来表示待计算的Tensor的各个元素的计算比例。(???存疑)
* 注意：使用`.backward()`方法的张量必须是标量（只有一个元素）


### `nn.Module`类

* 传给`nn.Conv2d()`的张量`size`应当为：$batchSize \times channels \times height \times width$
* `nn.Conv2d()`的输入输出的图像尺寸的关系为：

    $$output = \frac{input-kernelSize+2\times paddings}{stride}+1$$

    常用设置：核尺寸$3$，步长$1$，$padding\;1$。

* 损失函数使用`nn.函数名`调用
* 优化方法（adam什么的）在`nn.optim`中。
* 一般的训练过程：
    ```python
    dataloader = torch.utils.data.dataloader.Dataloader(traindata, batchsize, shuffle)
    for i in range(epoch):
        for j, (x,y) in enumerate(dataloader):
            pred = model(x)
            loss = lossFunc(pred, y)    # 计算loss
            optim.zero_grad()   # 将网络中的所有梯度初始化，这一步必须在backward之前。
            loss.backward() # 计算所有参数的梯度（仅计算不更新）
            optim.step()    # 更新梯度
    ```
    （实际训练过程中根据实际情况进行改动）

#### `nn.Module`中的网络层

* `nn.Linear(m, n)`线性全连接层。接受一个张量，输出一个张量，输入张量的`size`必须为`(*,m)`，输出的`size`为`(*,n)`。即只对最后一个维度进行全连接计算，再将各个维度拼接起来。这是为了保证在网络中进行随机梯度下降时(假设`batchSize = b`)，最后传到全连接层的张量`size`为`(b,m)`，这样设计可以保证全连接层的输出`size`为`(b,n)`，即只对每个样本进行计算，而不会把不同样本之间的数据放在一起计算。
* 使用`1x1`卷积，相当于在网络中使用一个全连接层，也可以用来压缩`channel`数。



## 使用`cuda`加速运算

* 使用`t.cuda.is_available()`判断`cuda`是否可用。
* 把模型/张量放到`gpu`计算：`x=x.cuda()`
* `torch.device("设备名")`定义了计算时使用的设备，例如`torch.device("cuda")`表示使用`cuda`，`torch.device("cpu")`表示使用`cpu`。使用`x.to(torch.device("设备名"))`可以将模型/张量放到相应的设备上。

## 数据加载

### Dataset

重写`torch.utils.data.dataset`类。该类必需的方法有：

* `__init__()`：初始化数据集，一般传入数据存放位置，存储标签信息文件路径等
* `__getitem__()`：定义给定一个索引，加载相应样本的方法。传入参是索引`i`，返回数据集中第`i`个样本的样本文件（`tensor`格式`PIL.image`格式(如果是图片的话)）以及该样本的标签。即：
```python
def __getitem__(self, index)
    [我是代码]
    return sample, label
```
* `__len__()`：返回数据集中的样本个数。
* 加载图片使用`PIL`模块中的`Image.open(path)`函数

### Dataloader

然后将`dataset`加载为`dataloader`：`torch.utils.data.dataloader.Dataloader(dataset, batch_size=n, shuffle=True, drop_last=True, num_worker)`
    * `batch_size`表示每个`batch`的大小。
    * `shuffle`表示是否在训练时打乱数据。
    * `drop_last`表示当最后剩下的数据不足一个batch时，是否丢弃。
    * `num_worker`表示**加载数据**使用的线程数。没错仅仅是加载数据的时候用一下多线程...

## 图像操作（预处理）

都在`torchvision.transforms`中。

* `Resize(h, w)`：将图片缩放为指定尺寸。只传入一个参数`x`时，将图像缩放使得其中一条边大小为`x`。
* `CenterCrop(h, w)`：从图片的中心裁剪下`h, w`大小。只传入一个参数`x`时，裁剪正方形。
* `RandomCrop(h, w)`：从图片中随机位置裁剪下`h, w`大小。只传入一个参数`x`时，裁剪正方形。
* `ToTesor()`：把`PIL_Image`或`numpy`数组变为`tensor`，同时进行归一化操作。
* `Normalize()`进行归一化操作。
* `ToPILImage(tensor)`：把`tensor`变为`PIL`格式
* `tensor.numpy()`：变为`numpy`数组
* 使用`matplotlib`显示图片：`plt.imshow(np.transpose(img, (1, 2, 0)))`，`img`是由`tensor`转化来的数组。之所以要转置，是因为`torch`和`numpy`中表示图片的格式不同（分别是`channel*h*w`和`h*w*channel`）
* 保存图像到本地：`torchvision.utils.save_image(img, path)`

## 训练网络的技巧

* 首先检查在训练集上的效果。如果训练集上效果就不好：
    * 试着换个损失函数/激活函数
    * 改变网络结构。很深的网络会导致靠前的层收到的梯度很小，学习很慢，导致梯度消失。可以调整学习率来应对。
    * 使用`maxout`网络，即让网络自己学习该用什么激活函数
    * 换一种优化方法
* 如果训练集效果好，测试集效果差（过拟合）：
    * 早停（当训练集正确率上升而测试集正确率下降时）
    * 使用`dropout`和`dropconnect`，即删除无用的部分
    * 加入正则项

## ~~论why有多蠢(番外篇)~~ why遇到的坑

### CudaRuntimeError

- 标签索引溢出：
    ```
    RuntimeError: cuda runtime error (59) : device-side assert triggered at
    /opt/conda/conda-bld/pytorch_1524586445097/work/aten/src/THC/generic/THCTensorCopy.c:70
    ```
    常常是因为`label`编号的最大值大于了总`label`数。大概率是因为网络最后的`FC`层输出没有和`label`数目匹配....换数据集一定要记得改全连接层啊啊啊啊