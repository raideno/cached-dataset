import os
import tqdm
import torch
import pathlib

def _save_sample_worker(args):
    dataset, base_path, index = args
    sample = dataset[index]
    file_path = os.path.join(base_path, f"{index}.pt")
    torch.save(sample, file_path)
    return index

def _filter_ds_store(file_name):
    return file_name != ".DS_Store"

def _better_listdir(path):
    return list(filter(_filter_ds_store, os.listdir(path)))

# SOURCE / INSPIRATION: https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/2
class DiskCachedDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that efficiently caches samples to disk for faster subsequent loading.

    This class is designed to store dataset samples as `.pt` files in a specified directory.
    If the dataset is not already cached, it will save the samples to disk during the first
    usage or on demand. Subsequent accesses load data directly from disk, improving
    performance when dealing with large datasets that do not fit into memory.

    Args:
        base_path (str): Path to the directory where the dataset is cached.
        transform (callable, optional): Transformation function applied to the dataset samples.
        verbose (bool, optional): Whether to display progress during dataset caching.

    Methods:
        cache_dataset: Saves the dataset samples to disk.
        verify_if_correctly_cached: Checks if the dataset is fully cached.
        get_missing_files: Identifies missing files in the cached dataset.
        is_there_missing_files: Returns whether any cached files are missing.
        load_dataset_or_cache_it: Loads the dataset from disk or caches it if needed.
        set_transform: Sets a new transformation function for the dataset.
    """
    def __init__(self, base_path, transform=None, verbose=False):
        self.base_path = base_path
        self.files_name = _better_listdir(self.base_path)
        self.transform = transform
        self.verbose = verbose
        
    def __len__(self):
        return len(self.files_name)
    
    def __getitem__(self, index):
        file_name = self.files_name[index]
        file_path = os.path.join(self.base_path, file_name)
        
        inputs, label = torch.load(file_path, weights_only=False)
        
        if self.transform is not None:
            inputs = self.transform(inputs)
            
        return inputs, label
        
    def set_transform(self, new_transform):
        self.transform = new_transform
        
    @staticmethod
    def cache_dataset(dataset, base_path, indices_to_cache=None, num_workers=0, verbose=False):
        if indices_to_cache is None:
            indices_to_cache = range(len(dataset))
            
        if num_workers > 0:
            import multiprocessing as mp
            
            args_list = [(dataset, base_path, index) for index in indices_to_cache]
            
            with mp.Pool(processes=num_workers) as pool:
                list(pool.imap(_save_sample_worker, args_list))
        else:
            for sample_index in tqdm.tqdm(total=len(dataset), iterable=indices_to_cache, desc="[caching-dataset]", disable=not verbose, initial=len(dataset) - len(indices_to_cache)):
                sample = dataset[sample_index]
                file_path = os.path.join(base_path, f"{sample_index}.pt")
                torch.save(sample, file_path)
        
        return DiskCachedDataset(base_path)
    
    @staticmethod
    def verify_if_correctly_cached(dataset, base_path) -> bool:
        files_names = _better_listdir(base_path)
        
        if len(files_names) != len(dataset):
            return False
        else:
            return True
       
    @staticmethod 
    def get_missing_files(dataset, base_path) -> tuple[bool, list[str], list[int]]:
        """
        Returns a tuple containing a boolean indicating whether there is missing files or not, the missing files names and their respective indexes.
        """
        all_possible_files = list(map(lambda index: f"{index}.pt", range(len(dataset))))
        existing_files = _better_listdir(base_path)
        
        missing_files = list(set(all_possible_files) - set(existing_files))
        
        missing_indexes = list(map(lambda file_name: int(file_name.split(".")[0]), missing_files))
        
        return len(missing_files) != 0, missing_files, missing_indexes
    
    @staticmethod
    def is_there_missing_files(dataset, base_path) -> bool:
        return DiskCachedDataset.get_missing_files(dataset, base_path)[0]
            
    @staticmethod
    def load_dataset_or_cache_it(dataset, base_path, num_workers=0, verbose=False):
        path_already_exist = os.path.exists(base_path)
        missing_indices = range(len(dataset)) if not path_already_exist else DiskCachedDataset.get_missing_files(dataset, base_path)[2]
        
        if not path_already_exist or len(missing_indices) > 0:
            pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
            
            return DiskCachedDataset.cache_dataset(
                dataset=dataset,
                base_path=base_path,
                num_workers=num_workers,
                indices_to_cache=missing_indices,
                verbose=verbose
            )
        else:
            return DiskCachedDataset(base_path)