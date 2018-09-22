---
title: why的python进阶教程！
date: 2018-09-17 09:53:07
tags:
- python 
- 编程
categories: 
- python
---

# why的python进阶教程！

## 算法&数据结构

#### 列表整体赋值

- 将列表中的元素分别赋值给多个变量：`a, b, c = list`，变量个数必须等于列表长度。
- 将部分列表赋值：`a, b, *x = list`，`c`会得到列表除去前两个元素剩下的部分。

<!--more-->

#### 保存循环中最近N轮的记录

- 使用`q = collextions.deque(maxlen = n)`，创建一个长度固定为`n`的列表，在使用`.append()`添加新元素时，旧元素会被挤出队列。
- 可以使用`q.pop()`手动弹出最右侧的元素，使用`p.popleft()`弹出最左侧的元素。
- 不指定大小时，`q = collextions.deque()`会创建一个可以无限增加新元素的队列。

#### 查找最大或最小的N个元素

- heapq模块有两个函数: nlargest() 和 nsmallest() 可以完美解决这个问题:
    - `headq.nlargest(n, list, key)`返回`list`中最大的`n`个元素，`nsmallest`同理。
    - `key`参数应当是一个函数，该函数接受`list`，并返回要排序的部分，通过灵活使用`key`可以对各种复杂数据结构进行排序。
- 内置的`sorted(list, key)`方法也可用来排序复杂结构，参数`key`含义同上。

#### 命名切片

- 可以提高代码可读性。`list[m:n:p]`可以写为：
    ```python 
    a = slice(m, n, p)
    list[a]
    ```
#### 查找序列中出现次数最多的元素

- 使用`c = collections.Counter(list)`将`list`包装为一个`Counter`，`Counter`包含了`list`中每个元素出现的次数。
- 调用`c.most_common(n)`可以返回`list`中出现次数最多的`n`个元素及其出现次数。
- 不止`list`，可以适用于任何可迭代对象。

#### 字典推导

- 可以快速创建字典，类似列表生成式：`{a:b for a, b in .... if ....}`
- 常用来快速创建一个字典的子集

#### 将字符串转化为字典

- 待转化的字符串必须是字典格式
- 使用`ast.literal_eval(str)`进行转化