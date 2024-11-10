from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Union, Optional
from packges.MyDatasets import load_dataset, batching
import numpy as np
import torch
import pandas as pd
import json

from packges.utils import get_model_weights, get_Pseudoinverse, permute_kv, Timers

# COSTANTS
SURPORTED_LOSS_FUNC = ["fro", "nuclear", "svd"] # "mse", "mae", "cos", "kl", 

now = datetime.now().strftime("%Y%m%d%H%M%S")
handler = RotatingFileHandler(f'/workspace/log/hf/hf_examples_{now}.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        handler,  # output to file
        logging.StreamHandler()  # output to console
    ]
)


def loss_fn(
    pred_kv,
    base_kv,
    loss_func: Union[str, list[str]] = "all",
    top_k: int = 0,
) -> dict:

    if loss_func == "all":
        loss_func = SURPORTED_LOSS_FUNC
    loss = {idx:[] for idx in loss_func}
    if "fro" in loss_func:          # Frobenius norm, compare every element in the matrix
        loss["fro"] = torch.norm(pred_kv - base_kv, p='fro').item()
    if "nuclear" in loss_func:    # nuclear norm, sum of the singular values of the matrix
        loss["nuclear"] = torch.norm((pred_kv - base_kv).squeeze(dim=0), p='nuc').item()
    # if "mse" in loss_func:
    #     loss["mse"] = torch.nn.MSELoss(pred_kv, base_kv).item()
    # if  "l1" in loss_func or  "mae" in loss_func:
    #     loss["mae"] = torch.nn.L1Loss(pred_kv, base_kv).item()
    # if "cos" in loss_func:    # cosine similarity,use when the matrix can be seen as vectors
    #     loss["cos"] = torch.nn.CosineSimilarity(pred_kv, base_kv).item()
    # if "kl" in loss_func:     # Kullback-Leibler divergence
    #     loss["kl"] = torch.nn.KLDivLoss(pred_kv, base_kv).item()
    if "svd" in loss_func:    # Singular Value Decomposition
        u1, s1, v1 = torch.svd(pred_kv)
        u2, s2, v2 = torch.svd(base_kv)
        if top_k == 0:
            loss["svd"] = torch.norm(s1 - s2, p='fro').item()
        else:
            logging.debug(f"s.shape : {s1.shape}")
            loss["svd"] = torch.norm(s1[:top_k] - s2[:top_k], p='fro').item()
    return loss

class KV_Pred_losses:
    def __init__(
        self,
        num_layers: int,
        loss_func: Union[str, list[str]] = "all",
    ):
        self.num_layers = num_layers
        if isinstance(loss_func, str):
            loss_func = [loss_func]
        self.loss_func = loss_func
        self.losses = {idx: [[],[]] for idx in self.loss_func}
    
    def reset(
        self,
    ):
        self.losses = [{idx: [] for idx in self.loss_func} * 2]
    
    def update(
        self,
        losses: dict[list],  # [k_loss, v_loss]
    ):
        for i in range(self.num_layers):
            for func in self.loss_func:
                self.losses[func][0].append(losses[func][0][i])
                self.losses[func][1].append(losses[func][1][i])
    
    def get_loss(
        self,
        mode: str = "avg",
    ):
        avg_loss = {idx: [[],[]] for idx in self.loss_func}
        if mode == "sum":
            return self.losses
        # avg mode default
        for i in range(self.num_layers):
            for func in self.loss_func:
                avg_loss[func][0].append(self.losses[func][0][i] / self.num_layers)
                avg_loss[func][1].append(self.losses[func][1][i] / self.num_layers)
        return avg_loss

def main(
    args: Union[None, Optional[dict]] = None
):
    assert args is not None, "args is None"
    
    aux_tokenizer = AutoTokenizer.from_pretrained(args["aux_name"])
    base_tokenizer = AutoTokenizer.from_pretrained(args["base_name"])
    aux = AutoModelForCausalLM.from_pretrained(args["aux_name"])
    base = AutoModelForCausalLM.from_pretrained(args["base_name"])
    dataset = load_dataset(args["dataset"])
    
    # logging.info(f"aux.config : {aux.config}")
    # logging.info(f"base.config : {base.config}")
    
    if args["batch_size"] > 0:
        dataset = batching(
            dataset=dataset,
            batch_size=args["batch_size"],
            shuffle=args["shuffle"],
            seed=args["seed"],
        )
        
    aux_weights = get_model_weights(aux, ["k_proj", "v_proj", "embed"])
    base_weights = get_model_weights(base, ["k_proj", "v_proj", "embed"])
    
    # for name, param in model_weights.items():
    #     print(name, param.shape)
    
    logging.debug(f"aux_k_proj : {aux_weights['k_proj'][0].shape}")
    logging.debug(f"base_k_proj : {base_weights['k_proj'][0].shape}")
    
    
    assert len(aux_weights["embed"]) == 1 and len(base_weights["embed"]) == 1, "aux or base model's embeddings length is not 1"
    
    aux_weights["k_proj"] = get_Pseudoinverse(aux_weights["k_proj"])
    aux_weights["v_proj"] = get_Pseudoinverse(aux_weights["v_proj"])
    
    aux_layers = len(aux_weights["k_proj"])
    base_layers = len(base_weights["k_proj"])
    
    losses = KV_Pred_losses(aux_layers, args["loss_func"])
    timer = Timers("request")
    timer.create_timer()
    
    assert aux_layers == base_layers, "aux and base model's layers are not the same"
    
    for idx, request in enumerate(dataset):
        timer.start()
        aux_request = aux_tokenizer(request["question"], return_tensors="pt")
        base_request = base_tokenizer(request["question"], return_tensors="pt")
        
        # TODO(zihao): align the input_ids and attention_mask more properly(simply drop the prefix now,Begining of the Sentence included, maybe hurt the performance)
        if aux_request['input_ids'].shape[1] < base_request['input_ids'].shape[1]:
            prefix_drop = base_request['input_ids'].shape[1] - aux_request['input_ids'].shape[1]
            base_request['input_ids'] = base_request['input_ids'][:, :-prefix_drop]
            base_request['attention_mask'] = base_request['attention_mask'][:, :-prefix_drop]
        elif aux_request['input_ids'].shape[1] > base_request['input_ids'].shape[1]:
            prefix_drop = aux_request['input_ids'].shape[1] - base_request['input_ids'].shape[1]
            aux_request['input_ids'] = aux_request['input_ids'][:, :-prefix_drop]
            aux_request['attention_mask'] = aux_request['attention_mask'][:, :-prefix_drop]
        
        logging.debug(f"aux_request : {aux_tokenizer.decode(aux_request['input_ids'][0])}")
        logging.debug(f"base_request : {base_tokenizer.decode(base_request['input_ids'][0])}")
        
        
        aux_output = aux(**aux_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        base_output = base(**base_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        
        aux_kv = aux_output.past_key_values
        base_kv = base_output.past_key_values
        
        aux_kv = permute_kv(aux_kv)
        base_kv = permute_kv(base_kv)
        
        logging.debug(f"aux_kv shape: {aux_kv[0][0].shape}")     # [batch_size, num_heads, num_tokens, head_dim]
        logging.debug(f"base_kv shape: {base_kv[0][0].shape}")   # to [batch_size, num_tokens, num_heads * head_dim]
        
        pred_kv = []
        
        loss = {idx:[[],[]] for idx in args["loss_func"]}
        
        for i in range(aux_layers):
            
            temp_k = aux_kv[i][0] @ aux_weights["k_proj"][i]
            temp_v = aux_kv[i][1] @ aux_weights["v_proj"][i]
            
            temp_k = temp_k @ base_weights["k_proj"][i]
            temp_v = temp_v @ base_weights["v_proj"][i]
            
            pred_kv.append([temp_k, temp_v])
            
            temp_loss = [loss_fn(temp_k, base_kv[i][0], loss_func=args["loss_func"]), loss_fn(temp_v, base_kv[i][1], loss_func=args["loss_func"])]
            
            for func in args["loss_func"]:
                loss[func][0].append(temp_loss[0][func])
                loss[func][1].append(temp_loss[1][func])
        
        timer.end()
        losses.update(loss)
        
        if idx % 200 == 0 or idx == len(dataset) - 1 :
            temp_loss = losses.get_loss(mode="avg")
            logging.info(f"request\t :\t {idx} \nloss func\t :\t {args['loss_func']} \ntime cost\t :\t {timer.get_time(mode='avg')}")
            logging.info(f"losses\t :\n{json.dumps(temp_loss, indent=4)}")
        
    logging.info(f"total time cost\t :\t {timer.get_time(mode='sum')}")
    final_loss = losses.get_loss(mode='avg')
    for func in args["loss_func"]:
        logging.info(f"total loss\t :\n{func} \nk_loss : {sum(final_loss[func][0])/len(final_loss[func][0])} \nv_loss : {sum(final_loss[func][1])/len(final_loss[func][1])}")
            

if __name__ == "__main__":
    args = {
        "aux_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_name": "meta-llama/Llama-2-7b-hf",
        "dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        "batch_size": 0,
        "shuffle": False,
        "seed": 42,
        "loss_func": "all",
    }
    if args["loss_func"] == "all":
        args["loss_func"] = SURPORTED_LOSS_FUNC
    logging.info(f"args : \n{args}")
    main(args)