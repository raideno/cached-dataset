import os
import torch
import pytest
import shutil
import tempfile

from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

def test_init_and_len(cached_dataset, mock_dataset):
    """Test initialization and length of cached dataset"""
    assert len(cached_dataset) == len(mock_dataset)

def test_getitem(cached_dataset, mock_dataset):
    """Test __getitem__ method of cached dataset"""
    for i in [0, 10, 50, 99]:  # Test a few indices
        cached_item, cached_label = cached_dataset[i]
        mock_item, mock_label = mock_dataset[i]
        
        print("item:", cached_item, mock_item)
        print("label:", cached_label, mock_label)
        
        assert cached_label == mock_label
        # TODO: fix it
        # assert torch.allclose(cached_item, mock_item)

def test_verify_if_correctly_cached(mock_dataset, temp_cache_dir, cached_dataset):
    """Test the verification method for complete caching"""
    assert DiskCachedDataset.verify_if_correctly_cached(mock_dataset, temp_cache_dir) == True
    
    # Remove a file to test incomplete caching
    os.remove(os.path.join(temp_cache_dir, "50.pt"))
    assert DiskCachedDataset.verify_if_correctly_cached(mock_dataset, temp_cache_dir) == False

def test_get_missing_files(mock_dataset, temp_cache_dir):
    """Test the missing files detection"""
    # Cache only a subset
    subset_indices = range(0, 90)
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=subset_indices
    )
    
    is_missing, missing_files, missing_indices = DiskCachedDataset.get_missing_files(
        mock_dataset, temp_cache_dir
    )
    
    assert is_missing == True
    assert len(missing_files) == 10
    assert all(idx >= 90 for idx in missing_indices)
    
    # Now cache the rest
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=missing_indices
    )
    
    is_missing_now, _, _ = DiskCachedDataset.get_missing_files(mock_dataset, temp_cache_dir)
    assert is_missing_now == False

def test_is_there_missing_files(mock_dataset, temp_cache_dir):
    """Test the is_there_missing_files method"""
    # First clear the directory
    for file in os.listdir(temp_cache_dir):
        os.remove(os.path.join(temp_cache_dir, file))
    
    assert DiskCachedDataset.is_there_missing_files(mock_dataset, temp_cache_dir) == True
    
    # Cache all files
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir
    )
    
    assert DiskCachedDataset.is_there_missing_files(mock_dataset, temp_cache_dir) == False

def test_load_dataset_or_cache_it(mock_dataset, temp_cache_dir):
    """Test the load_or_cache method"""
    # Clear directory first
    for file in os.listdir(temp_cache_dir):
        os.remove(os.path.join(temp_cache_dir, file))
    
    # Should create new cache
    dataset = DiskCachedDataset.load_dataset_or_cache_it(
        dataset=mock_dataset,
        base_path=temp_cache_dir
    )
    
    assert len(dataset) == len(mock_dataset)
    assert len(os.listdir(temp_cache_dir)) == len(mock_dataset)
    
    # Test with transform
    def simple_transform(x):
        data, label = x
        return (data * 2, label)
    
    dataset_with_transform = DiskCachedDataset.load_dataset_or_cache_it(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        transform=simple_transform
    )
    
    # The original data from mock_dataset
    orig_data, label = mock_dataset[0]
    # The transformed data
    transformed_data, _ = dataset_with_transform[0]
    
    assert torch.allclose(transformed_data, orig_data * 2)
    
def test_directory_creation(mock_dataset):
    """Test that the cache directory is created if it doesn't exist"""
    # Create a path that definitely doesn't exist
    temp_dir = tempfile.mkdtemp()
    non_existent_path = os.path.join(temp_dir, "non_existent_subdir")
    
    # Clean up the temp dir when we're done
    try:
        # This should create the directory and cache the dataset
        DiskCachedDataset.load_dataset_or_cache_it(
            dataset=mock_dataset,
            base_path=non_existent_path
        )
        
        # Verify the directory was created
        assert os.path.exists(non_existent_path)
        assert os.path.isdir(non_existent_path)
        
        # Verify files were created
        assert len(os.listdir(non_existent_path)) == len(mock_dataset)
    finally:
        shutil.rmtree(temp_dir)
        
def test_file_naming_and_sorting(mock_dataset, temp_cache_dir):
    """Test that files are named correctly and indices are properly sorted"""
    # Cache only a few indices in non-sequential order
    indices_to_cache = [5, 2, 9, 1]
    
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=indices_to_cache
    )
    
    # Check that the files exist with the correct names
    for idx in indices_to_cache:
        assert os.path.exists(os.path.join(temp_cache_dir, f"{idx}.pt"))
    
    # Create a DiskCachedDataset
    dataset = DiskCachedDataset(base_path=temp_cache_dir)
    
    # The length should match the number of files we created
    assert len(dataset) == len(indices_to_cache)
    
    # Access the items by index - the mapping should be consistent
    for i, idx in enumerate(sorted(indices_to_cache)):
        # The dataset now has files named 1.pt, 2.pt, 5.pt, 9.pt
        # We need to verify that the internal mapping is correct
        data, label = dataset[i]
        expected_data, expected_label = mock_dataset[idx]
        
        assert label == expected_label
        assert torch.allclose(data, expected_data)