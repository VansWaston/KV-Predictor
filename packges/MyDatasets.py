import pyarrow.parquet as pq
import pandas as pd
import os
import json
import random
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from logging.handlers import RotatingFileHandler
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    handler = RotatingFileHandler('/workspace/log/utils/datasets.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # handler,  # output to file
            logging.StreamHandler()  # output to console
        ]
    )

# load dataset via torch.utils.data.Dataset
class loading_dataset(Dataset):
    def __init__(self, data_file, max_dataset_size :int = None, use_prompt : bool = False):
        self.max_dataset_size = max_dataset_size
        self.data = self.load_data(data_file, use_prompt=use_prompt)
        logging.info(f"Loaded {len(self.data) if self.max_dataset_size is None else self.max_dataset_size} data from {data_file},with {len(self.data)} records in total")
    
    # HACK(zihao): load data with needed format
    def load_data(
        self, 
        data_file, 
        query_prompt = f"[INST] Answer the question directly within 5 words. Do NOT repeat the question or output any other words. Question: [/INST] ",
        use_prompt : bool = False,
    ) -> dict:
        Data = {}
        raw_data = self.load_raw_dataset(data_file, return_type="Dict")
        # adjust the format of the data
        if self.max_dataset_size is not None:
            for idx, record in enumerate(raw_data):
                if idx >= self.max_dataset_size:
                    break
                Data[idx] = {
                        'question': (query_prompt if use_prompt else "") + record["question"],
                        'answer': record["answer"]["value"],
                    }
        else:
            for idx, record in enumerate(raw_data):
                Data[idx] = {
                        'question': (query_prompt if use_prompt else "") + record["question"],
                        'answer': record["answer"]["normalized_value"],
                    }
        
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_raw_dataset(self, file_path, return_type="Dict"):
        '''
        read json file and return a {return_type}-typed object.  
        example:
        >>> dataset=load_dataset("path/to/dataset.json")    # DataFrame

        >>> dataset=dataset.to_dict(orient='records')
        >>> print(dataset[:5])
        '''
        if file_path.startswith("/"):
            file_path = file_path[1:]
        # Load the dataset.
        try:
            dataset = pd.read_json(f"file://localhost/{file_path}", lines=True)
        except ValueError as e:
            print(f"Error loading JSON: {e}")
            return None
        # Return the dataset.
        if return_type == "Dict":
            return dataset.to_dict(orient='records')
        elif return_type == "DataFrame":
            return dataset
        else:
            raise ValueError(f"Invalid return type: {return_type}, [Dict, DataFrame] supported.")

class datasets():
    def __init__(
        self,
        dir: str,
        file_type: str,
        file_name: str = None,
        save_dir: str = None,
        mode: str = None,
    )-> None:
        self.dir = dir
        self.file_type = file_type
        # if file_name is not specified, we will read all files matching the file_type in the directory
        if file_name is None:
            self.file_name = [file for file in os.listdir(dir) if file.endswith(file_type)]
        else:
            self.file_name = [file_name]
        
        if isinstance(mode, str):
            self.mode = mode
        else:
            self.mode = ['train', 'validation', 'test']
            
        self.data, loaded = self.read_data()
        self.save_dir = save_dir
        
        logging.info(f"Searched {len(self.file_name)} {file_type}_typed files from {dir}.Loaded {loaded} ones.")
        
    def read_data(self):
        data = pd.DataFrame()
        loaded = 0
        for file in self.file_name:
            if self.file_type == 'parquet' and self.mode in file:
                table = pq.read_table(os.path.join(self.dir, file)).to_pandas()
                data = pd.concat([data, table], ignore_index=True)
                logging.info(f"Loaded {file}")
                loaded += 1
            elif self.file_type == 'csv' and self.mode in file:
                table = pd.read_csv(os.path.join(self.dir, file))
                data = pd.concat([data, table], ignore_index=True)
                logging.info(f"Loaded {file}")
                loaded += 1
            else:
                pass
            
        return data, loaded
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data.iloc[idx]
    
    def save_json(
        self, 
        file_name: str, 
        lines: bool = False, 
        indent: int = None
    ):
        
        json_data = self.data.to_json(orient='records', lines=lines, indent=indent)
        with open(file_name, 'w') as f:
            f.write(json_data)
        logging.info(f"Saved data to {file_name}")

    def delete(
        self,
        columns: List[str] = None,
    ):
        if columns is not None:
            self.data = self.data.drop(columns=columns)
        else:
            logging.warning("No columns specified, deleting all data")
            self.data = pd.DataFrame()
        logging.info(f"Deleted data of {len(columns)} columns")

# decapricated: use collect_fn instead
def batching(
    dataset: Union[List,Tuple],
    batch_size: int,
    shuffle: bool = False,
    seed: int = None,
) -> List:
    """
    Batch dataset into limit batch_size
    Args:
        dataset (Union[List,Tuple]): dataset to be batched
        batch_size (int): required batch size
        shuffle (bool, optional): whether to shuffle, not if defaulted. Defaults to False.
        seed (int, optional): random seed. Defaults to None.

    Returns:
        List: dataset batched into limit batch_size
    """
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(dataset)
    batched = []
    if isinstance(dataset, Tuple):
        dataset = list(dataset)
    for i in range(0, len(dataset), batch_size):
        if i+batch_size < len(dataset):
            batched.append(dataset[i:i+batch_size])
        else:
            batched.append(dataset[i:])
    return batched

# HACK(zihao): collect_fn for DataLoader, to be used in DataLoader(collate_fn=collect_fn), *placed in main.py preferably*
def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    batched_data = {}
    query_prompt = f"\n\nAnswer the question within 5 words. Do NOT repeat the question. Do NOT output any other words. Question: "
    for sample in batch_samples:
        batch_inputs.append(query_prompt + sample['question'])
        batch_targets.append(sample['answer'])
    batched_data['question'] = batch_inputs
    batched_data['answer'] = batch_targets
    return batched_data


if __name__ == "__main__":
    folder = 'rc.nocontext'
    dir = f'/datasets/mandarjoshi/trivia_qa'
    file_type = 'parquet'
    mode = 'train'
    save_dir = dir
    
    data = datasets(f"{dir}/{folder}", file_type, save_dir=save_dir, mode=mode)
    
    data.delete(columns=['question_id','question_source','entity_pages','search_results'])
    
    print(data.data.head())
    data.save_json(f"{dir}/{folder}/{folder.replace('.','_')}_{mode}.json", lines=True, indent=0)
    # batched = batching(np.array(data.data).tolist(), 3, shuffle=True, seed=42)