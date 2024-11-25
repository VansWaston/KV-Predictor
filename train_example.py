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

from packges.predictor import KVPredictor
import torch.optim as optim
from packges.utils import constants
from packges.losses import CustomLoss

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
    aux = AutoModelForCausalLM.from_pretrained(args["aux_name"], device_map="auto", torch_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args["base_name"], device_map="auto", torch_dtype=torch.bfloat16)
    aux_tokenizer.pad_token = aux_tokenizer.eos_token
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    dataset = loading_dataset(args["dataset"])
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=args["shuffle"], collate_fn=collote_fn)
    
    logging.debug(f"aux.config : {aux.config}")
    logging.debug(f"base.config : {base.config}")
        
    aux_weights = get_model_weights(aux, ["k_proj", "v_proj", "embed"])
    base_weights = get_model_weights(base, ["k_proj", "v_proj", "embed"])
    
    # for name, param in model_weights.items():
    #     print(name, param.shape)
    
    logging.debug(f"aux_k_proj : {aux_weights['k_proj'][0].shape}")
    logging.debug(f"base_k_proj : {base_weights['k_proj'][0].shape}")
    
    assert len(aux_weights["embed"]) == 1 and len(base_weights["embed"]) == 1, "aux or base model's embeddings length is not 1"
    
    # aux_weights["k_proj"] = get_Pseudoinverse(aux_weights["k_proj"])        #  round 1e-4 with torch.dist(torch.pinverse @ matrix, torch.eye)
    # aux_weights["v_proj"] = get_Pseudoinverse(aux_weights["v_proj"])
    
    aux_layers = len(aux_weights["k_proj"])
    base_layers = len(base_weights["k_proj"])
    
    losses = constants()
    losses.register("loss")
    timer = Timers("request")
    timer.create_timer("total")
    timer.create_timer("train")
    timer.create_timer("prepare")
    
    model = KVPredictor(aux_layers, base_layers, aux.config.num_key_value_heads, base.config.num_key_value_heads, aux.config.head_dim, base.config.head_dim)
    model.train()
    model.to(args['device'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CustomLoss()
    idx = 0
    for batched_data in tqdm(dataloader):
        timer.start("total")
        timer.start("prepare")
        aux_request = aux_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=32).to(args['device'])
        base_request = base_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=32).to(args['device'])
        
        # TODO(zihao): align the input_ids and attention_mask more properly(simply drop the prefix now,Begining of the Sentence included, maybe hurt the performance)
        # NOTICE(zihao): can set to pad to max length!
        # align the requests
        # aux_request, base_request = align_requests(aux_request, base_request, aux_tokenizer, base_tokenizer, "cpu", "cpu", use_pad=args["use_pad"])
        
        aux_output = aux(**aux_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        base_output = base(**base_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        timer.end("prepare")
        
        aux_kv = aux_output.past_key_values
        base_kv = base_output.past_key_values
        
        # training
        timer.start("train")
        optimizer.zero_grad()  # 清除梯度
        pred_kv = model(aux_kv)  # 前向传播
        
        logging.debug(f"pred_kv shape: {pred_kv[0][0].shape}")
        logging.debug(f"base_kv shape: {base_kv[0][0].shape}")

        loss = loss_fn(pred_kv, base_kv)  # 计算损失
        
        logging.debug(f"loss : {loss}")
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        timer.end("train")
        
        # record = [aux_kv, base_kv]
        
        logging.debug(f"aux_kv shape: {aux_kv[0][0].shape}")     # [batch_size, num_heads, num_tokens, head_dim]
        logging.debug(f"base_kv shape: {base_kv[0][0].shape}")   # to [batch_size, num_tokens, num_heads * head_dim]
                        
        timer.end("total")
        losses.update("loss",loss)
        
        # save to json files
        # HACK(zihao): can save to parquet files for smaller size
        # record_serializable = [[[k.tolist(), v.tolist()] for k, v in record[0]], [[k.tolist(), v.tolist()] for k, v in record[1]]]
        # records.append(len(json.dumps(record_serializable, indent=4).encode("utf-8")) / 1024 / 1024)
        
        if idx % 200 == 0 or idx == len(dataloader) - 1 :
            logging.info(f"batch {idx}\n"
                         f"---time cost:---\n"
                         f"total: {timer.get_time('total')}, train: {timer.get_time('train')}, prepare: {timer.get_time('prepare')}\n"
                         f"---loss---\n"
                         f"loss: {losses.report('loss')}\n"
                         )
            
            # logging.info(f"losses\t :\n{json.dumps(temp_loss, indent=4)}")
            # logging.info(f"KV file memory usage\t :\t {sum(records)/len(records)} MB")
        
        idx += 1
        
    logging.info(f"total time cost\t :\t {timer.get_time('total', mode='sum')}\n"
                 f"train time cost\t :\t {timer.get_time('train', mode='sum')}\n"
                 f"prepare time cost\t :\t {timer.get_time('prepare', mode='sum')}\n"
                 f"losses\t :\n{losses.report('loss')}\n"
                 )


if __name__ == "__main__":
    args = {
        "aux_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_name": "meta-llama/Llama-2-7b-hf",
        "dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        "use_pad": True,
        "batch_size": 16,
        "shuffle": True,
        "seed": 42,
        "device": "cuda",
        "loss_func": "all",
    }
    if args["loss_func"] == "all":
        args["loss_func"] = SURPORTED_LOSS_FUNC
    logging.info(f"args : \n{args}")
    main(args)