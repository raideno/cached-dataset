import os
import torch
import tempfile

from tests.helpers import temp_cache_dir, cached_dataset, mock_dataset
from cached_dataset.dataset import _cache_single_sample, _filter_ds_store, _better_listdir

def test_filter_ds_store():
    """Test the DS_Store filtering function"""
    assert _filter_ds_store(".DS_Store") == False
    assert _filter_ds_store("file.txt") == True
    assert _filter_ds_store("1.pt") == True

def test_better_listdir(temp_cache_dir):
    """Test the improved listdir function"""
    # Create some files including .DS_Store
    files = ["file1.txt", "file2.txt", ".DS_Store"]
    for file in files:
        with open(os.path.join(temp_cache_dir, file), 'w') as f:
            f.write("")
    
    result = _better_listdir(temp_cache_dir)
    assert ".DS_Store" not in result
    assert "file1.txt" in result
    assert "file2.txt" in result
    assert len(result) == 2

def test_cache_single_sample(mock_dataset, temp_cache_dir):
    """Test the single sample caching function"""
    index = 42
    args = (mock_dataset, temp_cache_dir, index)
    returned_index = _cache_single_sample(args)
    
    # Check that the function returns the correct index
    assert returned_index == index
    
    # Check that the file was created
    assert os.path.exists(os.path.join(temp_cache_dir, f"{index}.pt"))
    
    # Load the file and verify its contents
    data = torch.load(os.path.join(temp_cache_dir, f"{index}.pt"))
    tensor, label = data
    
    assert label == index % 5  # According to our mock dataset
    assert tensor[0].item() == float(index)
    assert tensor[1].item() == float(index + 1)