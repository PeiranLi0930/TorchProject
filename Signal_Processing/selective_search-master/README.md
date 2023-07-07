# Selective Search

English | [简体中文](README_CN.md)

[![GitHub release](https://img.shields.io/github/v/release/ChenjieXu/selective_search?include_prereleases)](https://github.com/ChenjieXu/selective_search/releases/)
[![PyPI](https://img.shields.io/pypi/v/selective_search)](https://pypi.org/project/selective-search/)
[![Conda](https://img.shields.io/conda/v/chenjiexu/selective_search)](https://anaconda.org/ChenjieXu/selective_search)

[![Travis Build Status](https://travis-ci.org/ChenjieXu/selective_search.svg?branch=master)](https://travis-ci.org/ChenjieXu/selective_search)
[![Codacy grade](https://img.shields.io/codacy/grade/8d5b9ce875004d458bdf570f4d719472)](https://www.codacy.com/manual/ChenjieXu/selective_search)

This is a complete implementation of selective search in Python. I thoroughly read the related
papers [[1]](#Uijlings)[[2]](#Felzenszwalb)[[3]](#koen) and the author’s MATLAB implementation. Compared with other
implementations, my method is authentically shows the idea of the original paper. Moreover, this method has clear logic
and rich annotations, which is very suitable for teaching purposes, allowing people who have just entered the CV field
to understand the basic principles of selective search and exercise code reading ability.

## Installation

Installing from [PyPI](https://pypi.org/project/selective-search/) is recommended :

```
$ pip install selective-search
```

It is also possible to install the latest version from [Github source](https://github.com/ChenjieXu/selective_search/):

```
$ git clone https://github.com/ChenjieXu/selective_search.git
$ cd selective_search
$ python setup.py install
```

Install from [Anaconda](https://anaconda.org/ChenjieXu/selective_search):

```bash
conda install -c chenjiexu selective_search
```

## Quick Start

```python
import skimage.io
from selective_search import selective_search

# Load image as NumPy array from image files
image = skimage.io.imread('path/to/image')

# Run selective search using single mode
boxes = selective_search(image, mode='single', random_sort=False)
```

For detailed examples, refer [this](https://github.com/ChenjieXu/selective_search/tree/master/examples) part of the
repository.

## Parameters

### Mode

Three modes correspond to various combinations of diversification strategies. The appoach to combine different
diversification strategies, say, color spaces, similarity measures, starting regions is listed in the following
table[[1]](#Uijlings).

| Mode    | Color Spaces        | Similarity Measures | Starting Regions (k) | Number of Combinations |
|---------|---------------------|---------------------|----------------------|------------------------|
| single  | HSV                 | CTSF                | 100                  | 1                      |
| fast    | HSV, Lab            | CTSF, TSF           | 50, 100              | 8                      |
| quality | HSV, Lab, rgI, H, I | CTSF, TSF, F, S     | 50, 100, 150, 300    | 80                     |

* **Color
  Space** [[Source Code]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/util.py#L23)  
  Initial oversegmentation algorithm and our subsequent grouping algorithm are performed in this colour space.

* **Similarity
  Measure** [[Source Code]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/measure.py#L101)  
  'CTSF' means the similarity measure is aggregate of color similarity, texture similarity, size similarity, and fill
  similarity.

* **Starting
  Region** [[Source Code]](https://github.com/ChenjieXu/selective_search/blob/master/selective_search/util.py#L9)  
  A parameter of initial grouping algorithm[[2]](#Felzenszwalb), which yields high quality starting locations
  efficiently. A larger k causes a preference for larger components of initial strating regions.

### Random Sort

If random_sort set to True, function will carry out pseudo random sorting. It only alters sequences of bounding boxes,
instead of locations, which prevents heavily emphasis on large regions as combing proposals from up to 80 different
strategies[[1]](#Uijlings). This only has a significant impact when selecting a subset of region proposals with high
rankings, as in RCNN.

## References

\[1\] <a name="Uijlings"> [J. R. R. Uijlings et al., Selective Search for Object Recognition, IJCV, 2013](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)  
\[2\] <a name="Felzenszwalb"> [Felzenszwalb, P. F. et al., Efficient Graph-based Image Segmentation, IJCV, 2004](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)  
\[3\] <a name='koen'> [Segmentation as Selective Search for Object Recognition](https://www.koen.me/research/selectivesearch/)