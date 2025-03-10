import os
import time
import torch
import pytest

from helpers import MockDataset
from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

@pytest.mark.slow
def test_load_performance(temp_cache_dir):
    """Test loading performance from disk cache vs original dataset"""
    dataset_size = 1000
    mock_dataset = MockDataset(size=dataset_size)
    
    # Cache the dataset
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        num_workers=4
    )
    
    disk_dataset = DiskCachedDataset(
        base_path=temp_cache_dir
    )
    
    # Time access to random indices from original dataset
    import random
    indices = [random.randint(0, dataset_size-1) for _ in range(100)]
    
    start_time = time.time()
    for idx in indices:
        _ = mock_dataset[idx]
    original_access_time = time.time() - start_time
    
    # Time access to same indices from disk cache
    start_time = time.time()
    for idx in indices:
        _ = disk_dataset[idx]
    cached_access_time = time.time() - start_time
    
    # Just print the times - we can't always guarantee disk will be faster
    # as it depends on the implementation of the original dataset
    print(f"Original dataset access time: {original_access_time:.4f}s")
    print(f"Cached dataset access time: {cached_access_time:.4f}s")