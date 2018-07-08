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

* nn.Conv2d()支持的张量格式为：$batchSize \times channels \times height \times width$
* 损失函数使用`nn.函数名`调用
* 优化方法（adam什么的）在`nn.optim`中。
* 一般的训练过程：
    ```python
    for i in range(epoch):
        for (x,y) in trainData:
            pred = model(x)
            loss = lossFunc(pred, y)    # 计算loss
            optim.zero_grad()   # 将网络中的所有梯度初始化，这一步必须在backward之前。
            loss.backward() # 计算所有参数的梯度（仅计算不更新）
            optim.step()    # 更新梯度
    ```
    （实际训练过程中根据实际情况进行改动）
* 
