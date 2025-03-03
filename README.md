# Cached Dataset

> 🚨 **WARNING: This package is still under development and NOT ready for production use!** 🚨

The idea is that when you have datasets with computation hungry transformations, you can wrap your dataset with the cached-dataset in order to cache the transformed version of your dataset either into disk or memory & thus avoid recomputing the transformations during each epoch of your training.

Depending on the context this can save a lot of time, but at the cost of memory consumption.

The package supports multi processing & is thus able to apply and cache your transformations as fast as possible.

## Installation

```
pip install git+https://github.com/raideno/cached-dataset.git
```
