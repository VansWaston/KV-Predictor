from transformers import AutoModelForCausalLM

from packges.decorators import timer

from typing import Union, Optional
import torch
import numpy as np
from time import time
import logging

def get_model_weights(
    model: AutoModelForCausalLM,
    weights_dict: list[str],
    transpose: bool = True,
):
    model_weights = model.state_dict()
    res = {}
    for attr in weights_dict:
        res[attr] = [model_weights[name].T for name in model_weights.keys() if attr in name] if transpose else [model_weights[name] for name in model_weights.keys() if attr in name]
    
    return res

@timer
def get_Pseudoinverse(
    matries: list[Union[np.ndarray, torch.Tensor]],
    device: str = None,
):
    if device is None:
        device = matries[0].device
    
    if isinstance(matries[0], np.ndarray):
        matries = [torch.tensor(matrix, device=device) for matrix in matries]
    elif isinstance(matries[0], torch.Tensor):
        matries = [matrix.to(device) for matrix in matries]
        
    assert all([matrix.device == device for matrix in matries]), f"matries are not in the same device : device : {device} , matries : {matries[0].device}"
    
    res = []
    for idx, matrix in enumerate(matries):
        # justify the matrix is invertible
        if matrix.shape[0] != matrix.shape[1]:
            logging.debug(f"matrix is not invertible : {matrix.shape}")
        elif torch.det(matrix) == 0:
            logging.debug(f"matrix is not invertible : torch.det(matrix) == 0")
        else:
            logging.debug(f"matrix is invertible {idx}/{len(matries)}")
            
        pinv = torch.pinverse(matrix)
        # torch.linalg.pinv
        logging.debug(f"pinv similarity : {torch.dist(torch.eye(pinv.shape[0]), pinv @ matrix)}")
        res.append(pinv)

    return res

def permute_kv(
    kvs,
):
    """
    transpose the key and value matrices to [batch_size, num_tokens, num_heads * head_dim]
    """
    num_layers = len(kvs)
    bsz, num_heads, num_tokens, head_dim = kvs[0][0].shape
    
    res = ()
    for i in range(num_layers):
        layer_kv_list = list(kvs[i])
        layer_kv_list[0] = layer_kv_list[0].permute(0, 2, 1, 3).reshape(bsz, num_tokens, -1)
        layer_kv_list[1] = layer_kv_list[1].permute(0, 2, 1, 3).reshape(bsz, num_tokens, -1)
        res += (tuple(layer_kv_list),)
    
    # logging.debug(f"permuted_kv.shape : {res[0][0].shape}")
    return res

# TODO(zihao): align the input_ids and attention_mask more properly(simply drop the prefix now,Begining of the Sentence included, maybe hurt the performance)
# consider to pad the shorter one with the eos_token_id
def align_requests(
    aux_request,
    base_request,
    aux_tokenizer,
    base_tokenizer,
    aux_device,
    base_device,
    use_pad = False,
):
    if not (aux_device == base_device):
        aux_request["input_ids"] = aux_request["input_ids"].cpu()
        aux_request["attention_mask"] = aux_request["attention_mask"].cpu()
        base_request["input_ids"] = base_request["input_ids"].cpu()
        base_request["attention_mask"] = base_request["attention_mask"].cpu()
    
    if aux_request['input_ids'].shape[1] < base_request['input_ids'].shape[1]:
        if use_pad:
            pad_id = torch.tensor([[aux_tokenizer.bos_token_id] * (base_request['input_ids'].shape[1] - aux_request['input_ids'].shape[1])] * aux_request['input_ids'].shape[0])
            aux_request["input_ids"] = torch.cat([pad_id, aux_request["input_ids"]], dim=-1)
            aux_request["attention_mask"] = torch.cat([torch.ones_like(pad_id), aux_request["attention_mask"]], dim=-1)
        else:
            prefix_drop = base_request['input_ids'].shape[1] - aux_request['input_ids'].shape[1]
            base_request['input_ids'] = base_request['input_ids'][:, :-prefix_drop]
            base_request['attention_mask'] = base_request['attention_mask'][:, :-prefix_drop]
    elif aux_request['input_ids'].shape[1] > base_request['input_ids'].shape[1]:
        if use_pad:
            pad_id = torch.tensor([[base_tokenizer.bos_token_id] * (aux_request['input_ids'].shape[1] - base_request['input_ids'].shape[1])] * base_request['input_ids'].shape[0])
            base_request["input_ids"] = torch.cat([pad_id, base_request["input_ids"]], dim=-1)
            base_request["attention_mask"] = torch.cat([torch.ones_like(pad_id), base_request["attention_mask"]], dim=-1)
        else:
            prefix_drop = aux_request['input_ids'].shape[1] - base_request['input_ids'].shape[1]
            aux_request['input_ids'] = aux_request['input_ids'][:, :-prefix_drop]
            aux_request['attention_mask'] = aux_request['attention_mask'][:, :-prefix_drop]
    
    if not (aux_device == base_device):
        aux_request["input_ids"] = aux_request["input_ids"].to(aux_device)
        aux_request["attention_mask"] = aux_request["attention_mask"].to(aux_device)
        base_request["input_ids"] = base_request["input_ids"].to(base_device)
        base_request["attention_mask"] = base_request["attention_mask"].to(base_device)
    return aux_request, base_request

class Timers:
    """
    Attributes:
    -----------
    classname : str
        The name of the timer instance.
    time : dict
        A dictionary to store the elapsed time for each timer.
    start_time : dict
        A dictionary to store the start time for each timer.
    Methods:
    --------
    create_timer(name: str = None):
        Creates a new timer with the given name. If no name is provided, uses the classname.
    reset(name: str = None):
        Resets the timer with the given name. If no name is provided, uses the classname.
    start(name: str = None):
        Starts the timer with the given name. If no name is provided, uses the classname.
    end(name: str = None):
        Ends the timer with the given name and records the elapsed time. If no name is provided, uses the classname.
    get_time(name: str = None, mode: str = "avg"):
        Returns the elapsed time for the timer with the given name.  If no name is provided, uses the classname.
        The mode can be "sum" to get the total time or "avg" to get the average time.
    """
    
    def __init__(
        self,
        name: str,
    ):
        self.classname = name
        self.time = {}
        self.start_time = {}
    
    def create_timer(
        self,
        name: str = None,
    ):
        if name is None:
            name = self.classname
        self.time[name] = []
        self.start_time[name] = 0
    
    def reset(
        self,
        name: str = None,
    ):
        if name is None:
            name = self.classname
        self.time[name] = []
        self.start_time[name] = 0
    
    def start(
        self,
        name: str = None,
    ):
        if name is None:
            name = self.classname
        self.start_time[name] = time()
    
    def end(
        self,
        name : str = None,
    ):
        if name is None:
            name = self.classname
        self.time[name].append(time() - self.start_time[name])
        
    def get_time(
        self,
        name: str = None,
        mode: str = "avg",
    ):
        if name is None:
            name = self.classname
        if mode == "sum":
            return sum(self.time[name])
        # average mode default
        return sum(self.time[name]) / len(self.time[name])
    
class constants():
    def __init__(self):
        self.names = []
        self.values = {}
        
    def register(self, name, value = None):
        self.names.append(name)
        if value is not None:
            self.values[name] = [value]
        else:
            self.values[name] = []

    def update(self, name, value):
        if name in self.names:
            self.values[name].append(value)
        else:
            self.register(name, value)
    
    def report(self, name, mode = "avg"):
        if mode == "sum":
            return sum(self.values[name])
        elif mode == "last":
            return self.values[name][-1]
        else:
            return sum(self.values[name]) / len(self.values[name])