---
title: whyの学习笔记：python网络爬虫
date: 2018-07-12 20:46:04
tags:
- 编程
- python
categories: python
---

# 使用Urllib

* **打开并读取一个`url`** ： `htmlName = urllib.request.urlopen('要打开的url').read().decode('编码方式')`：
    * 返回值是一个`html`格式的对象
    * 为了正确显示中文，一般都在后面加上`.decode('utf-8')`
* **下载文件** ： 使用`urllib.request.urlretrieve(DOWNLOAD_URL, './img/image1.png')`，`DOWNLOAD_URL`是网络上的文件地址，第二个参数是本地的存储地址。

<!--more-->

# 美丽の汤（BeautifulSoup）

* 是一种代替正则表达式检索网页内容的工具。导入：`from bs4 import BeautifulSoup`。
* 把一个`html`对象（就是`urlopen`返回的那个对象）变成美丽の汤（包装成一个**soup类**）：`soup = BeautifulSoup(html, features='lxml')`，`features`表示解析方法，一般就用`lxml`。
* **寻找某个特定标签（`tag`）名：**`soup.find('a')`，返回值是tag名`a`第一次出现的`tag`，美味の汤会把这个`tag`包装为`soup`类（而不是`tag`类，和下面作区分）
* **寻找所有特定的标签（`tag`）名**：`result = soup.find_all('a')`，返回所有标签名是`a`的`tag`。
    * 返回值是一个可迭代对象，它的每一个元素是一个**tag对象**。
    * 一个`tag`对象包含多个属性。可以直接检索包含特定属性内容的特定`tag`：`result = soup.find_all('tag名', {'属性名:'用来匹配属性值的正则表达式'})`。例如，想要找到所有叫做`img`的`tag`，并且要求这些`tag`的`src`属性的末尾是`.jpg`，可以使用`result = soup.find_all('img', {'src:'.*\.jpg'})`。
* 一个`tag`对象包含多个属性。**检索特定属性**信息方法和字典相同：`tag['属性名']`，一般比较关心该`tag`指向的链接，那么就用`tag['herf']`。
    * 也可以直接通过`tag.get_text()`获取该`tag`的正文内容（如果有的话）

# requests模块

### 使用`requests`访问网站

* 网络数据收发方式：
    * get：获取网页内容。用户发送的get请求一般会直接显示在`url`里。
    * post：把本地的数据传到服务器（例如账号登录，上传文件），用户`post`的数据不会显示在`url`中。
* **使用`requests`发送`get`请求**：`requests.get('要get的url', params=用字典表示的参数)`。
    * 返回值是符合`get`请求的一个 **`Response`类**（与之前`urlopen`返回的的`html`对象、美丽汤返回的`soup`类似） 。
    * 例如调用百度搜索可以`requests.get('http://www.baidu.com/s', params={"wd": "搜索内容"})`，就可以返回搜索页面对应的 **`Response`类** 。
* **`post`数据到服务端（账户登录等）**：`r = requests.post('要post'的url,data=要post的信息)`
    * 要post的信息用字典表示，例如`{'username':'why', 'passwod':'233'}`
    * 返回值`r`也是一个 **`response`类** 。
    * 以账户登录为例，登录之后会生成一个该账户登录状态的`cookie`，使用`r.cookies`获取
    * 在向已登录状态的网页发送`get`请求，需要用到之前`post`得到的`cookie`：`requests.get('要get的url', cookies=r.cookies)`
* 可以使用`r.content`来访问 **`Response`类** 存储的内容。
* **使用会话**：使用会话（`session`）可以免去每次`get`都要传递`cookie`：
    * 创建一个`session`：`s = requests.Session()`
    * 使用`session`进行`post`：`r = s.post(url, data=data)`
    * `post`一次以后，就可以多次使用`cookie`进行`get`：`s.get(url)`，无需手动传入`cookie`。

### 使用`requests`下载数据

使用`requests.get`获取文件的内容，然后新建一个本地文件，将文件内容写入。
```python
r = requests.get(IMAGE_URL)
with open('./img/image2.png', 'wb') as f:
    f.write(r.content)
```
其中`stream=False`表示先把整个文件下载到内存，然后再写入文件。如果想要实时下载并写入：

```python
r = requests.get(IMAGE_URL, stream=True)    # stream loading

with open('./img/image3.png', 'wb') as f:
    for chunk in r.iter_content(chunk_size=32):
        f.write(chunk)
```
`chunk=32`表示将要下载的文件分成多个区块，每个大小为`32byte`。

# 网络爬虫

### 爬取数据的一般步骤：

* 进入网页：一般使用`urlopen`函数
* 搜索数据：使用`BeautifulSoup`从`html`中锁定`tag`，然后用正则表达式搜索想要的内容。总之就是美味汤和正则混合使用...
* 下载数据：使用`urllib`或者`requests`

## 高级爬虫`Selenium`

可以使用浏览器自动完成操作。需要安装：`python`库`Selenium`，相应的浏览器插件`KATALON Recorder`（用来记录操作和生成`python`代码）

* 只要把生成的代码复制以后执行即可，需要导入的内容：
    ```python
    from selenium import webdriver
    driver = webdriver.Chrome()     # 也可以是别的浏览器
    [复制来的代码]
    ```
* 使用`html = driver.page_source`来获取网页的`html`
* 可以使用`driver.get_screenshot_as_file"./img/sreenshot1.png")`来截图
* 最后要`driver.close()`关闭浏览器
* 如果让浏览器在后台执行，作如下修改：
    ```python
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_argument("--headless")       # define headless

    driver = webdriver.Chrome(chrome_options=chrome_options)
    ```

## 多进程爬虫

与普通的多进程程序一样，把要多进程加速的函数放入进程池`pool`，唯一要注意的是为了防止多个进程爬取到重复的网页，可以定义集合（`set`）来存储已经爬取过的网页(`seenUrl`)和未爬取的网页(`unseenUrl`)。每当某个进程爬取到一个网页之后，就从`unseenUrl`中删除该网页，向`seenUrl`中加入该网页，所有爬虫都从`unseenUrl`中读取网页。

## 爬虫框架`Scrapy`

* 新建一个`Scrapy`项目：    
    ```shell
    scrapy startproject <项目名>
    ```
* 在项目中新建一个爬虫（`spider`），会建立一个爬虫的模板：
    ```shell
    scrapy genspider <爬虫名字> <要爬的域名>
    ```
* 使用爬虫进行爬取：
    ```shell
    scrapy crawl <爬虫名字>
    ```
* `爬虫名字.py`进行数据的爬取，`items.py`定义爬取到数据的结构，`pipelines.py`定义处理爬取到的`items`的方法。
* 使用交互环境`scrapy shell`： 
    ```shell
    scrapy shell "url"
    ```
    打开包含该`url`的`Response`的`scrapy shell`


### `item`类

一般写在`items.py`中，用来保存爬取到的数据的结构，类似字典。格式为：
```python
class myItem(scrapy.item):
    成员1 = scrapy.Field()
    成员2 = scrapy.Field()
    ......
```


### `scrapy.spider.Spider`类

需要包含以下成员：

* `name`: 爬虫的名字，必须是唯一的
* `allowed_domains`：允许爬取的域名列表
* `start_urls`：url列表，在找不到后续连接时，从该列表中爬取

可以有以下方法：

* `start_requests`（不必要）：用来创建用于爬取的`request`对象，可以不写，默认是从`start_urls`中的`url`自动创建。如果希望修改最初的爬取对象，需要重写该方法。
* `parse`（必要）：接受并解析`response`对象（类似之前`requests`中的`response`对象），从中提取出`item`，并进一步生成要跟进的`url`的`request`对象。
    * 该方法是一个生成器，通过`yeild`返回爬取的数据（`item`）
    * 要解析的网页数据在`response.body`中。
    * 一般使用`response.xpath("xpath地址")`来筛选元素，返回符合条件的`tag`列表。`xpath`简单语法如下所示：
        
        |xpath|含义|
        |:-:|:-:|
        |`/html/head/title` | 选择`<HTML>`文档中 `<head>` 标签内的 `<title>` 元素|
        |`/html/head/title/text()` | 选择上面提到的 `<title>` 元素的文字|
        |`//td` |选择所有的 `<td>` 元素|
        |`//div[@class="mine"]` | 选择所有具有 `class="mine"` 属性的 `div` 元素|




执行过程：

* 为`start_urls`中的`url`创建相应的`request`对象
* `request`对象创建`response`对象，并传给`parse`方法进行解析
* `parse`将解析得到的`item`形式的数据返回。




### `scrapy.contrib.spiders.CrawlSpider`类

一种`scrapy`内置的常用爬虫，是`spider`的子类。有额外的属性`rules`：包含爬取规则的列表，每个元素时一个`rule`对象，列表越靠前的规则优先级越高。对于一个`rule`对象，包含：
    
* `link_extractor`：是一个 `Link Extractor` 对象。 其定义了如何从爬取到的页面提取链接。












