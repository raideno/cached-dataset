import os
import tqdm
import torch
import pathlib

FILE_EXTENSION = "pt"

def _cache_single_sample(args: tuple[torch.utils.data.Dataset, str, int]) -> int:
    dataset, base_path, sample_index = args
    
    sample = dataset[sample_index]
    
    file_path = os.path.join(base_path, f"{sample_index}.{FILE_EXTENSION}")
    
    torch.save(sample, file_path)
    
    return sample_index

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
        transform (callable, optional): Transformation function applied to the dataset samples. Default is `None`.
        verbose (bool, optional): Whether to display progress during dataset caching. Default is `False`.

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
        self.transform = transform
        self.verbose = verbose
        
        self.files_names = _better_listdir(self.base_path)
        
    def __len__(self):
        return len(self.files_names)
    
    def __getitem__(self, index):
        file_name = self.files_names[index]
        file_path = os.path.join(self.base_path, file_name)
        
        sample = torch.load(file_path, weights_only=False)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
        
    @staticmethod
    def cache_dataset(dataset, base_path, indices_to_cache=None, num_workers=0, verbose=False):
        """
        Caches the specified dataset samples to disk as `.pt` files.

        This method iterates through the dataset (or a subset of it) and saves each sample 
        as a `.pt` file in the given `base_path`. It supports multiprocessing for faster caching.

        Args:
            dataset (torch.utils.data.Dataset): The dataset containing samples to cache.
            base_path (str): Path to the directory where the dataset should be cached.
            indices_to_cache (iterable, optional): Specific indices of samples to cache. 
                If `None`, all samples are cached. Default is `None`.
            num_workers (int, optional): Number of worker processes to use for caching.
                If set to 0, caching is done sequentially. Default is `0`.
            verbose (bool, optional): Whether to display a progress bar during caching. Default is `False`.

        Returns:
            bool: `True` when caching completes successfully.

        Example:
            >>> DiskCachedDataset.cache_dataset(dataset=dataset, base_path="/cache_path", num_workers=4, verbose=True)
        """
        if indices_to_cache is None:
            indices_to_cache = range(len(dataset))
            
        if num_workers > 0:
            import multiprocessing
            
            arguments_list: tuple[torch.utils.data.Dataset, str, int] = [(dataset, base_path, index) for index in indices_to_cache]
            
            progress_bar = tqdm.tqdm(total=len(dataset), desc="[caching-dataset]", disable=not verbose, initial=len(dataset) - len(indices_to_cache))

            with multiprocessing.Pool(processes=num_workers) as pool:
                for _ in pool.imap_unordered(_cache_single_sample, arguments_list, chunksize=max(1, len(arguments_list) // (num_workers * 4))):
                    if verbose:
                        progress_bar.update(1)
                        
            if verbose:
                progress_bar.close()
        else:
            for sample_index in tqdm.tqdm(total=len(dataset), iterable=indices_to_cache, desc="[caching-dataset]", disable=not verbose, initial=len(dataset) - len(indices_to_cache)):
                sample = dataset[sample_index]
                file_path = os.path.join(base_path, f"{sample_index}.{FILE_EXTENSION}")
                torch.save(sample, file_path)
        
        return True
    
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
        Identifies missing cached files in the specified directory.

        This method compares the expected dataset samples (based on dataset length) 
        with the files present in the `base_path` directory. It returns information 
        about any missing files.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to compare against.
            base_path (str): Path to the directory where the dataset should be cached.

        Returns:
            tuple[bool, list[str], list[int]]: 
                - **bool**: `True` if there are missing files, `False` otherwise.
                - **list[str]**: A list of missing file names (e.g., `"3.pt"`).
                - **list[int]**: A list of corresponding missing sample indices.

        Example:
            >>> missing, missing_files, missing_indices = DiskCachedDataset.get_missing_files(dataset, "/cache_path")
            >>> if missing:
            ...     print(f"[warning]: missing files: {missing_files}")
        """
        all_possible_files = list(map(lambda index: f"{index}.{FILE_EXTENSION}", range(len(dataset))))
        existing_files = _better_listdir(base_path)
        
        missing_files = list(set(all_possible_files) - set(existing_files))
        
        missing_indexes = list(map(lambda file_name: int(file_name.split(".")[0]), missing_files))
        
        return len(missing_files) != 0, missing_files, missing_indexes
    
    @staticmethod
    def is_there_missing_files(dataset, base_path) -> bool:
        """
        Checks if there are missing cached files for the dataset.

        This method verifies whether the dataset is fully cached by checking for 
        missing sample files in the specified `base_path`.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to compare against.
            base_path (str): Path to the directory where the dataset should be cached.

        Returns:
            bool: `True` if there are missing files, `False` otherwise.

        Example:
            >>> if DiskCachedDataset.is_there_missing_files(dataset, "/cache_path"):
            ...     print("[warning]: some dataset samples are missing.")
        """
        return DiskCachedDataset.get_missing_files(dataset, base_path)[0]
            
    @staticmethod
    def load_dataset_or_cache_it(dataset, base_path, transform=None, verbose=False, num_workers=0):
        """
        Loads a cached dataset from disk or caches it if missing files are detected.

        This method checks if the dataset is already cached in the specified `base_path`. 
        If the cache is incomplete or does not exist, it caches the missing samples.
        The returned `DiskCachedDataset` instance can then be used for data loading.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to cache or load.
            base_path (str): Path to the directory where the dataset should be cached.
            transform (callable, optional): Transformation function to apply to dataset samples.
            verbose (bool, optional): Whether to display progress during caching. Default is `False`.
            num_workers (int, optional): Number of worker processes to use for caching. Default is `0`.

        Returns:
            DiskCachedDataset: An instance of the `DiskCachedDataset` class for loading data.

        Example:
            >>> dataset = DiskCachedDataset.load_dataset_or_cache_it(
            ...     dataset=dataset, 
            ...     base_path="/cache_path", 
            ...     transform=my_transform, 
            ...     verbose=True, 
            ...     num_workers=4
            ... )
        """
        path_already_exist = os.path.exists(base_path)
        missing_indices = range(len(dataset)) if not path_already_exist else DiskCachedDataset.get_missing_files(dataset, base_path)[2]
        
        if not path_already_exist or len(missing_indices) > 0:
            pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
            
            DiskCachedDataset.cache_dataset(
                dataset=dataset,
                base_path=base_path,
                num_workers=num_workers,
                indices_to_cache=missing_indices,
                verbose=verbose
            )
            
            return DiskCachedDataset(
                base_path=base_path,
                transform=transform,
                verbose=verbose
            )
        else:
            return DiskCachedDataset(
                base_path=base_path,
                transform=transform,
                verbose=verbose
            )