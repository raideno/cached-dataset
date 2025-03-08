import os
import torch
import shutil
import pytest
import tempfile

from torch.utils.data import Dataset

class MockDataset(Dataset):
    """A simple dataset for testing"""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return torch.tensor([float(idx), float(idx+1)]), idx % 5

@pytest.fixture
def mock_dataset():
    return MockDataset(size=100)

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for testing caching"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def cached_dataset(mock_dataset, temp_cache_dir):
    """Return a pre-cached dataset for tests that need it"""
    from cached_dataset import DiskCachedDataset
    
    DiskCachedDataset.cache_dataset(
        dataset=mock_dataset,
        base_path=temp_cache_dir,
        verbose=False
    )
    
    return DiskCachedDataset(
        base_path=temp_cache_dir,
        verbose=False
    )