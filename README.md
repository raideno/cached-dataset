# Cached Dataset

> 🚨 **WARNING: This package is still under development and NOT ready for production use!** 🚨

The idea is that when you have datasets with computation hungry transformations, you can wrap your dataset with the cached-dataset in order to cache the transformed version of your dataset either into disk or memory & thus avoid recomputing the transformations during each epoch of your training.

Depending on the context this can save a lot of time, but at the cost of memory consumption.

The package supports multi processing & is thus able to apply and cache your transformations as fast as possible.

## Installation

```
pip install git+https://github.com/raideno/cached-dataset.git
```

## Usage

```python
from cached_dataset import DiskCachedDataset

# NOTE: your usual torch dataset with transforms for which you want to cache the transformed version
dataset = ...

# NOTE: the directory were you want to cache your dataset.
CACHING_DIRECTORY = "./cached-dataset"

cached_dataset = DiskCachedDataset.load_dataset_or_cache_it(
    dataset=dataset,
    base_path=CACHING_DIRECTORY,
    verbose=True
)

for sample in cached_dataset:
    print(f"[sample-{i}]: {sample}")
```

**Note:** for now the only available caching location is on disk, memory isn't supported yet.
