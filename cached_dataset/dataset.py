import os
import tqdm
import torch
import pathlib

from typing import Callable

# NOTE: (already_cached_len, total_len)
Callback = Callable[[tuple[int, int]], None]

def _save_sample_worker(args):
    dataset, base_path, callback, index = args
    sample = dataset[index]
    file_path = os.path.join(base_path, f"{index}.pt")
    torch.save(sample, file_path)
    if callback is not None:
        # TODO: fix as it should be (index, len(indices_to_cache)) to be more accurate
        callback((index, len(dataset)))
    return index

def _filter_ds_store(file_name):
    return file_name != ".DS_Store"

def _better_listdir(path):
    return list(filter(_filter_ds_store, os.listdir(path)))

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# SOURCE / INSPIRATION: https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/2
class DiskCachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, preload=False, item_size_estimator=None, item_size_estimator_batch_size=16, device=None, transform=None, verbose=False):
        """"
        Warning: setting preload=True will try to load all the dataset into memory. Depending on the dataset size and your computer's memory size this might or might not be possible and can lead to the crash of your notebook.
        Note: when item_size_estimator is None, the DiskCachedDataset will attempt to use some predefined / popular item_size_estimator if possible and will throw an error if it is not possible to do so.
        Note: the item_size_estimator will be un on multiple items and the average will be taken to get a better estimate of the item size. THe number of items on which the item_size_estimator will be run is determined by the item_size_estimator_batch_size parameter.
        """
        self.base_path = base_path
        self.files_name = _better_listdir(self.base_path)
        self.transform = transform
        self.device = device if device is not None else DEFAULT_DEVICE
        self.preload = preload
        self.verbose = verbose
        
        self.items = None
        self.preloaded = False
        
        if self.preload:
            self.items = self.__preload_items()
            self.preloaded = True
        
    def __len__(self):
        return len(self.files_name)
    
    # QUESTION: should we move the elements to the device memory like now or is it a bad idea and it would be better to move them in the training loop ?
    # What is the impact ?
    
    # TODO: when computing the memory, do so in the correct device
    # TODO: do something to pre-check the computer memory size and tell whether it is possible to load the dataset or not 
    def __preload_items(self):
        items = []
        
        with tqdm.tqdm(iterable=self.files_name, desc="[preloading-dataset]", disable=not self.verbose) as progress_bar:
            for file_name in progress_bar:
                file_path = os.path.join(self.base_path, file_name)
                item = torch.load(file_path, map_location=self.device, weights_only=False)
                items.append(item)
            
        return items
        
    def __getitem__(self, index):
        if self.preload and self.preloaded:
            inputs, label = self.items[index]
        else:
            file_name = self.files_name[index]
            file_path = os.path.join(self.base_path, file_name)
            
            inputs, label = torch.load(file_path, weights_only=False)
            
            if self.transform is not None:
                inputs = self.transform(inputs)
            
        return inputs, label
        
    def set_transform(self, new_transform):
        self.transform = new_transform
        
    @staticmethod
    def cache_dataset(dataset, base_path, indices_to_cache=None, num_workers=0, callback:Callback=None):
        if indices_to_cache is None:
            indices_to_cache = range(len(dataset))
            
        if num_workers > 0:
            import multiprocessing as mp
            
            args_list = [(dataset, base_path, callback, index) for index in indices_to_cache]
            
            with mp.Pool(processes=num_workers) as pool:
                list(pool.imap(_save_sample_worker, args_list))
        else:
            for i, sample_index in enumerate(indices_to_cache):
                sample = dataset[sample_index]
                file_path = os.path.join(base_path, f"{sample_index}.pt")
                torch.save(sample, file_path)
                if callback is not None:
                    already_cached_len = len(dataset) - len(indices_to_cache) - (i + 1)
                    total_len = len(dataset)
                    
                    callback((already_cached_len, total_len))
        
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
    def load_dataset_or_cache_it(dataset, base_path, num_workers=0, callback:Callback=None):
        path_already_exist = os.path.exists(base_path)
        missing_indices = range(len(dataset)) if not path_already_exist else DiskCachedDataset.get_missing_files(dataset, base_path)[2]
        
        if not path_already_exist or len(missing_indices) > 0:
            pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
            
            return DiskCachedDataset.cache_dataset(
                dataset=dataset,
                base_path=base_path,
                num_workers=num_workers,
                callback=callback,
                indices_to_cache=missing_indices
            )
        else:
            return DiskCachedDataset(base_path)