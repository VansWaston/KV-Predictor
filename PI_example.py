from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Union, Optional
from packges.MyDatasets import loading_dataset, collote_fn
import numpy as np
import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from packges.utils import get_model_weights, get_Pseudoinverse, permute_kv, Timers, align_requests
from packges.losses import KV_Pred_losses, SURPORTED_LOSS_FUNC, loss_fn

now = datetime.now().strftime("%Y%m%d%H%M%S")
handler = RotatingFileHandler(f'/workspace/log/examples/PI_examples_{now}.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        handler,  # output to file
        logging.StreamHandler()  # output to console
    ]
)


def main(
    args: Union[None, Optional[dict]] = None
):
    assert args is not None, "args is None"
    
    aux_tokenizer = AutoTokenizer.from_pretrained(args["aux_name"])
    base_tokenizer = AutoTokenizer.from_pretrained(args["base_name"])
    aux = AutoModelForCausalLM.from_pretrained(args["aux_name"])
    base = AutoModelForCausalLM.from_pretrained(args["base_name"])
    aux_tokenizer.pad_token = aux_tokenizer.eos_token
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    dataset = loading_dataset(args["dataset"])
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=args["shuffle"], collate_fn=collote_fn)
    # logging.info(f"aux.config : {aux.config}")
    # logging.info(f"base.config : {base.config}")
    
        
    aux_weights = get_model_weights(aux, ["k_proj", "v_proj", "embed"])
    base_weights = get_model_weights(base, ["k_proj", "v_proj", "embed"])
    
    # for name, param in model_weights.items():
    #     print(name, param.shape)
    
    logging.debug(f"aux_k_proj : {aux_weights['k_proj'][0].shape}")
    logging.debug(f"base_k_proj : {base_weights['k_proj'][0].shape}")
    
    
    assert len(aux_weights["embed"]) == 1 and len(base_weights["embed"]) == 1, "aux or base model's embeddings length is not 1"
    
    aux_weights["k_proj"] = get_Pseudoinverse(aux_weights["k_proj"])        #  round 1e-4 with torch.dist(torch.pinverse @ matrix, torch.eye)
    aux_weights["v_proj"] = get_Pseudoinverse(aux_weights["v_proj"])
    
    aux_layers = len(aux_weights["k_proj"])
    base_layers = len(base_weights["k_proj"])
    
    losses = KV_Pred_losses(aux_layers, args["loss_func"])
    timer = Timers("request")
    timer.create_timer()
    records = []
    
    assert aux_layers == base_layers, "aux and base model's layers are not the same"
    idx = 0
    for batched_data in tqdm(dataloader):
        timer.start()
        aux_request = aux_tokenizer(batched_data["question"], return_tensors="pt", padding=True, truncation=True)
        base_request = base_tokenizer(batched_data["question"], return_tensors="pt", padding=True, truncation=True)
        
        # TODO(zihao): align the input_ids and attention_mask more properly(simply drop the prefix now,Begining of the Sentence included, maybe hurt the performance)
        # align the requests
        aux_request, base_request = align_requests(aux_request, base_request, aux_tokenizer, base_tokenizer, "cpu", "cpu", use_pad=args["use_pad"])
        
        logging.debug(f"aux_request : {aux_tokenizer.decode(aux_request['input_ids'][0])}")
        logging.debug(f"base_request : {base_tokenizer.decode(base_request['input_ids'][0])}")
        
        
        aux_output = aux(**aux_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        base_output = base(**base_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        
        aux_kv = aux_output.past_key_values
        base_kv = base_output.past_key_values
        
        aux_kv = permute_kv(aux_kv)
        base_kv = permute_kv(base_kv)
        
        record = [aux_kv, base_kv]
        
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
        
        record_serializable = [[[k.tolist(), v.tolist()] for k, v in record[0]], [[k.tolist(), v.tolist()] for k, v in record[1]]]
        records.append(len(json.dumps(record_serializable, indent=4).encode("utf-8")) / 1024 / 1024)
        
        if idx % 200 == 0 or idx == len(dataset) - 1 :
            temp_loss = losses.get_loss(mode="avg")
            
            logging.info(f"request batch\t :\t {idx} \nloss func\t :\t {args['loss_func']} \ntime cost\t :\t {timer.get_time(mode='avg')} seconds")
            # logging.info(f"losses\t :\n{json.dumps(temp_loss, indent=4)}")
            logging.info(f"KV file memory usage\t :\t {sum(records)/len(records)} MB")
        
        idx += 1
        
    logging.info(f"total time cost\t :\t {timer.get_time(mode='sum')}")
    final_loss = losses.get_loss(mode='avg')
    for func in args["loss_func"]:
        logging.info(f"total loss\t :\n{func} \nk_loss : {sum(final_loss[func][0])/len(final_loss[func][0])} \nv_loss : {sum(final_loss[func][1])/len(final_loss[func][1])}")


if __name__ == "__main__":
    args = {
        "aux_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_name": "meta-llama/Llama-2-7b-hf",
        "dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        "use_pad": True,
        "batch_size": 1,
        "shuffle": True,
        "seed": 42,
        "loss_func": "all",
    }
    if args["loss_func"] == "all":
        args["loss_func"] = SURPORTED_LOSS_FUNC
    logging.info(f"args : \n{args}")
    main(args)