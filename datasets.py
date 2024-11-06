import pyarrow.parquet as pq
import pandas as pd
import os
import json
import random
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # handler,  # output to file
        logging.StreamHandler()  # output to console
    ]
)


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
        
        if mode == None:
            self.mode = ['train', 'validation', 'test']
        else:
            self.mode = mode
            
        self.data = self.read_data()
        self.save_dir = save_dir
        
        logging.info(f"Loaded {len(self.file_name)} {file_type}_typed files from {dir}")
        
    def read_data(self):
        data = pd.DataFrame()
        for file in self.file_name:
            if self.file_type == 'parquet' and self.mode in file:
                table = pq.read_table(os.path.join(self.dir, file)).to_pandas()
                data = pd.concat([data, table], ignore_index=True)
            elif self.file_type == 'csv' and self.mode in file:
                table = pd.read_csv(os.path.join(self.dir, file))
                data = pd.concat([data, table], ignore_index=True)
            else:
                pass
            logging.info(f"Loaded {file}")
        return data
    
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
    
def batching(
    dataset: Union[List,Tuple],
    batch_size: int,
    shuffle: bool = False,
    seed: int = None,
):
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

if __name__ == "__main__":
    folder = 'rc.nocontext'
    dir = f'/home/zihao/datasets/mandarjoshi/trivia_qa'
    file_type = 'parquet'
    mode = 'validation'
    save_dir = dir
    
    data = datasets(f"{dir}/{folder}", file_type, save_dir=save_dir, mode=mode)
    
    data.delete(columns=['question_id','question_source','entity_pages','search_results'])
    
    print(data.data.head())
    data.save_json(f"{dir}/{folder}/{folder.replace('.','_')}_{mode}.json", lines=True, indent=0)
    # batched = batching(np.array(data.data).tolist(), 3, shuffle=True, seed=42)