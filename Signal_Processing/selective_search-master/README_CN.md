# Selective Search

简体中文 | [English](README.md)

[![GitHub release](https://img.shields.io/github/v/release/ChenjieXu/selective_search?include_prereleases)](https://github.com/ChenjieXu/selective_search/releases/)
[![PyPI](https://img.shields.io/pypi/v/selective_search)](https://pypi.org/project/selective-search/)
[![Conda](https://img.shields.io/conda/v/chenjiexu/selective_search)](https://anaconda.org/ChenjieXu/selective_search)

[![Travis Build Status](https://travis-ci.org/ChenjieXu/selective_search.svg?branch=master)](https://travis-ci.org/ChenjieXu/selective_search)
[![Codacy grade](https://img.shields.io/codacy/grade/8d5b9ce875004d458bdf570f4d719472)](https://www.codacy.com/manual/ChenjieXu/selective_search)

这是选择性搜索用Python的完整实现。 我仔细阅读了相关论文[[1]](#Uijlings)[[2]](#Felzenszwalb)[[3]](#koen)和作者MATLAB实现，相较于其他实现，本方法原汁原味的展示了原论文的思想。
而且本方法逻辑清晰，注释丰富，非常适合作为教学目的，让刚刚进入视觉领域的人了解选择性搜索的基本原理并锻炼代码阅读能力。

## 安装

推荐从 [PyPI](https://pypi.org/project/selective-search/) 安装:

```
$ pip install selective-search
```

从 [Github源](https://github.com/ChenjieXu/selective_search/) 安装最新版本也是可选的:

```
$ git clone https://github.com/ChenjieXu/selective_search.git
$ cd selective_search
$ python setup.py install
```

从[Anaconda](https://anaconda.org/ChenjieXu/selective_search) 安装:

```bash
conda install -c chenjiexu selective_search
```

## 快速上手

```python
import skimage.io
from selective_search import selective_search

# 加载本地图片为numpy向量形式
image = skimage.io.imread('path/to/image')

# 使用单一模式运行选择性搜索
boxes = selective_search(image, mode='single', random_sort=False)
```

更详细的例子请参考仓库的 [这个部分](https://github.com/ChenjieXu/selective_search/tree/master/examples) 。

## 参数

### 模式

三种模式对应多样化策略的各种组合。 下表列出了将不同的多样化策略（例如色彩空间，相似性度量，起始区域）组合在一起的方法。

| 模式    | 色彩空间       | 相似性度量 | 起始区域 (k) | 组合数量 |
|---------|---------------------|---------------------|----------------------|------------------------|
| 单一   | HSV                 | CTSF                | 100                  | 1                      |
| 快速    | HSV, Lab            | CTSF, TSF           | 50, 100              | 8                      |
| 质量 | HSV, Lab, rgI, H, I | CTSF, TSF, F, S     | 50, 100, 150, 300    | 80                     |

* **色彩空间** [[源码]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/util.py#L23)  
  最初的超分割算法和我们随后的分组算法都是在此色彩空间中执行的。

* **相似性度量** [[源码]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/measure.py#L101)  
  “CTSF”表示相似性度量是颜色相似性，纹理相似性，大小相似性和填充相似性的集合。

* **起始区域** [[源码]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/util.py#L9)  
  初始分组算法的参数[[2]](#Felzenszwalb), 有效地产生高质量的起始位置。 较大的k会导致分割倾向于较大的初始条纹区域。

### 随机排序

如果把随机排序设置为真， 函数将执行伪随机排序。 它仅更改边界框的顺序，而不是位置，这避免了在梳理多达80种不同策略的提案时过分强调大区域[[1]](#Uijlings)。
如RCNN中的，这仅在选择具有较高排名的区域提案的子集时才产生重大影响。

## 参考

\[1\] <a name="Uijlings"> [J. R. R. Uijlings et al., Selective Search for Object Recognition, IJCV, 2013](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)  
\[2\] <a name="Felzenszwalb"> [Felzenszwalb, P. F. et al., Efficient Graph-based Image Segmentation, IJCV, 2004](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)  
\[3\] <a name='koen'> [Segmentation as Selective Search for Object Recognition](https://www.koen.me/research/selectivesearch/)
