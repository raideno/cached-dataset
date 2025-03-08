import os
import torch
import pytest
import shutil
import tempfile

from torch.utils.data import Dataset
from cached_dataset.dataset import DiskCachedDataset
from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset

class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("Dataset is empty")

class SingleItemDataset(Dataset):
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Dataset has only one item")
        return torch.tensor([1.0, 2.0]), 0

@pytest.fixture
def empty_dataset():
    return EmptyDataset()

@pytest.fixture
def single_item_dataset():
    return SingleItemDataset()

def test_empty_dataset(empty_dataset, temp_cache_dir):
    """Test behavior with an empty dataset"""
    # Should not throw errors
    DiskCachedDataset.cache_dataset(
        dataset=empty_dataset,
        base_path=temp_cache_dir
    )
    
    # Verify no files were created
    assert len(os.listdir(temp_cache_dir)) == 0
    
    # Verify correctly cached returns True (technically it is correct that there are 0 files)
    assert DiskCachedDataset.verify_if_correctly_cached(empty_dataset, temp_cache_dir) == True
    
    # No missing files
    is_missing, missing_files, missing_indices = DiskCachedDataset.get_missing_files(
        empty_dataset, temp_cache_dir
    )
    assert is_missing == False
    assert len(missing_files) == 0
    assert len(missing_indices) == 0

def test_single_item_dataset(single_item_dataset, temp_cache_dir):
    """Test behavior with a dataset containing only one item"""
    DiskCachedDataset.cache_dataset(
        dataset=single_item_dataset,
        base_path=temp_cache_dir
    )
    
    # Verify one file was created
    assert len(os.listdir(temp_cache_dir)) == 1
    assert os.path.exists(os.path.join(temp_cache_dir, "0.pt"))
    
    # Create dataset and verify it works
    dataset = DiskCachedDataset(base_path=temp_cache_dir)
    assert len(dataset) == 1
    
    data, label = dataset[0]
    assert label == 0
    assert torch.allclose(data, torch.tensor([1.0, 2.0]))

def test_corrupt_cache_file(mock_dataset, temp_cache_dir):
    """Test behavior with corrupt cache files"""
    # Cache the dataset
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=[0, 1, 2]
    )
    
    # Corrupt one of the files
    with open(os.path.join(temp_cache_dir, "1.pt"), 'w') as f:
        f.write("This is not a valid PyTorch file")
    
    # Initialize the dataset
    dataset = DiskCachedDataset(base_path=temp_cache_dir)
    
    # Should be able to access uncorrupted files
    data0, label0 = dataset[0]
    assert label0 == 0
    
    # Accessing the corrupted file should raise an error
    with pytest.raises(Exception):
        dataset[1]
        
    # Test partial recaching with corrupt files
    # First, identify missing files (which should include the corrupt one)
    missing_indices = DiskCachedDataset.get_missing_files(mock_dataset, temp_cache_dir)[2]
    
    # Recache the corrupt file by index
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        indices_to_cache=[1]
    )
    
    # Now should be able to access the fixed file
    dataset = DiskCachedDataset(base_path=temp_cache_dir)
    data1, label1 = dataset[1]
    assert label1 == 1