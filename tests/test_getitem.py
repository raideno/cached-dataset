import torch

from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

def test_getitem_with_slice(cached_dataset, mock_dataset):
    """Test __getitem__ with slices in cached dataset"""
    # Example: Testing slice [0:5] with step 1
    indices = list(range(0, 5))
    cached_items = cached_dataset[0:5]
    
    # Check if length of sliced result is correct
    assert len(cached_items) == 5
    
    for idx, cached_item in zip(indices, cached_items):
        mock_item, mock_label = mock_dataset[idx]
        assert torch.allclose(cached_item[0], mock_item)
        assert cached_item[1] == mock_label


def test_getitem_with_large_slice(cached_dataset, mock_dataset):
    """Test __getitem__ with a larger slice in cached dataset"""
    # Example: Testing slice [0:20] with step 1
    indices = list(range(0, 20))
    cached_items = cached_dataset[0:20]
    
    # Check if length of sliced result is correct
    assert len(cached_items) == 20
    
    for idx, cached_item in zip(indices, cached_items):
        mock_item, mock_label = mock_dataset[idx]
        assert torch.allclose(cached_item[0], mock_item)
        assert cached_item[1] == mock_label


def test_getitem_with_step_slice(cached_dataset, mock_dataset):
    """Test __getitem__ with step in slices in cached dataset"""
    # Example: Testing slice [0:10:2] with step 2
    indices = list(range(0, 10, 2))
    cached_items = cached_dataset[0:10:2]
    
    # Check if length of sliced result is correct
    assert len(cached_items) == 5  # There should be 5 items in this range
    
    for idx, cached_item in zip(indices, cached_items):
        mock_item, mock_label = mock_dataset[idx]
        assert torch.allclose(cached_item[0], mock_item)
        assert cached_item[1] == mock_label