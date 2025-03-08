import os
import torch
import pytest

from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
def test_cache_dataset_with_workers(mock_dataset, temp_cache_dir, num_workers):
    """Test caching with different numbers of workers"""
    # Clear directory first
    for file in os.listdir(temp_cache_dir):
        os.remove(os.path.join(temp_cache_dir, file))
    
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        num_workers=num_workers,
        verbose=True  # Test that verbose mode doesn't break
    )
    
    # Verify all files are created
    assert len(os.listdir(temp_cache_dir)) == len(mock_dataset)
    
    # Check content of a few files
    for idx in [0, 25, 50, 75, 99]:
        file_path = os.path.join(temp_cache_dir, f"{idx}.pt")
        assert os.path.exists(file_path)
        
        data = torch.load(file_path)
        tensor, label = data
        
        # Verify against mock dataset
        mock_tensor, mock_label = mock_dataset[idx]
        assert label == mock_label
        assert torch.allclose(tensor, mock_tensor)

def test_partial_caching_with_workers(mock_dataset, temp_cache_dir):
    """Test caching only a subset of indices with workers"""
    # Clear directory first
    for file in os.listdir(temp_cache_dir):
        os.remove(os.path.join(temp_cache_dir, file))
    
    # Cache only even indices
    even_indices = list(range(0, 100, 2))
    
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=even_indices,
        num_workers=2
    )
    
    # Verify only even files are created
    files = os.listdir(temp_cache_dir)
    assert len(files) == len(even_indices)
    
    for idx in range(100):
        file_path = os.path.join(temp_cache_dir, f"{idx}.pt")
        if idx % 2 == 0:
            assert os.path.exists(file_path)
        else:
            assert not os.path.exists(file_path)
    
    # Now cache the odd indices
    odd_indices = list(range(1, 100, 2))
    
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=odd_indices,
        num_workers=2
    )
    
    # Verify all files are now created
    assert len(os.listdir(temp_cache_dir)) == len(mock_dataset)