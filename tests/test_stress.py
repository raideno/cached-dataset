import os
import time
import torch
import pytest

from helpers import MockDataset
from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

@pytest.mark.slow
def test_large_dataset_performance(temp_cache_dir):
    """Test performance with a large dataset"""
    # Create a larger dataset
    large_dataset = MockDataset(size=10000)
    
    # Time the caching operation
    start_time = time.time()
    DiskCachedDataset.cache_dataset(
        dataset=large_dataset,
        base_path=temp_cache_dir,
        num_workers=0,
        verbose=True
    )
    single_process_time = time.time() - start_time
    
    # Clear the cache directory
    for file in os.listdir(temp_cache_dir):
        os.remove(os.path.join(temp_cache_dir, file))
    
    # Time with multiple workers
    start_time = time.time()
    DiskCachedDataset.cache_dataset(
        dataset=large_dataset,
        base_path=temp_cache_dir,
        num_workers=4,
        verbose=True
    )
    multi_process_time = time.time() - start_time
    
    # Check all files were created
    assert len(os.listdir(temp_cache_dir)) == len(large_dataset)
    
    # Assert that multiprocessing is faster (allows up to 10% margin for overhead)
    # Skip this assertion if running in CI environment where resources might be limited
    if not os.environ.get('CI'):
        assert multi_process_time < single_process_time, \
            f"Multiprocessing ({multi_process_time:.2f}s) should be faster than single process ({single_process_time:.2f}s)"

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