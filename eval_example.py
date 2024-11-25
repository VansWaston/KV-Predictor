from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Union, Optional, Tuple
from packges.MyDatasets import loading_dataset, collote_fn
import numpy as np
import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from rouge import Rouge

from packges.utils import get_model_weights, get_Pseudoinverse, permute_kv, Timers, align_requests
from packges.losses import KV_Pred_losses, SURPORTED_LOSS_FUNC, loss_fn
from packges.MyGenerate import mygenerate, mygenerate_deprecated

now = datetime.now().strftime("%Y%m%d%H%M%S")
handler = RotatingFileHandler(f'/workspace/log/examples/eval_examples_{now}.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
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
    
    if "device" in args and isinstance(args["device"], str):
        args["device"] = [args["device"], args["device"]]
    device = args["device"] if "device" in args else ["cuda:0","cuda:1"] if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ["cpu","cpu"]
    logging.info(f"device : {device}")

    dataset = loading_dataset(args["dataset"], use_prompt=args["use_prompt"])
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=args["shuffle"], collate_fn=collote_fn)
    
    aux_tokenizer = AutoTokenizer.from_pretrained(args["aux_name"])
    base_tokenizer = AutoTokenizer.from_pretrained(args["base_name"])
    aux = AutoModelForCausalLM.from_pretrained(args["aux_name"], device_map=device[0])
    base = AutoModelForCausalLM.from_pretrained(args["base_name"], device_map=device[1])
    aux.eval()
    base.eval()
    aux_tokenizer.pad_token = aux_tokenizer.eos_token
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # logging.info(f"aux.config : {aux.config}")
    # logging.info(f"base.config : {base.config}")
        
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
    
    timer = Timers("request")
    timer.create_timer("total")
    timer.create_timer("base")
    
    assert aux_layers == base_layers, "situations that aux and base model's layers are not the same are not supported"
    
    preds , refs = [], []
    rouge = Rouge()
    for batched_data in tqdm(dataloader):
        timer.start("total")
        aux_request = aux_tokenizer(batched_data["question"], return_tensors="pt", padding=args["tokenizer_padding"], truncation=args["tokenizer_truncation"], 
                                    padding_side=args["tokenizer_padding_side"]).to(device[0])
        base_request = base_tokenizer(batched_data["question"], return_tensors="pt", padding=args["tokenizer_padding"], truncation=args["tokenizer_truncation"], 
                                    padding_side=args["tokenizer_padding_side"]).to(device[1])
        
        base_refs = base_tokenizer(batched_data["answer"], return_tensors="pt", padding=True, truncation=True)
        base_refs = base_tokenizer.batch_decode(base_refs["input_ids"], skip_special_tokens=True)
        
        # align the requests
        logging.debug(f"token length is {'' if  ['input_ids'].shape[1]==base_request['input_ids'].shape[1] else 'not'} the same, using aligning...")
        aux_request, base_request = align_requests(aux_request, base_request, aux_tokenizer, base_tokenizer, device[0], device[1], use_pad=args["use_pad"])
        
        logging.debug(f"aux_request['input_ids'].length after adjusting : {aux_request['input_ids'].shape}")
        logging.debug(f"base_request['input_ids'].length after adjusting : {base_request['input_ids'].shape}")
        
        # generate the key and value matrices for the auxiliary model
        # thus only run the auxiliary model **once**
        aux_output = aux(**aux_request, output_hidden_states=True, output_attentions=True ,use_cache=True)
        
        aux_kv = aux_output.past_key_values
        aux_kv = permute_kv(aux_kv)
        
        logging.debug(f"aux_kv shape: {aux_kv[0][0].shape}")     # [batch_size, num_heads, num_tokens, head_dim]
        
        pred_kv = ()
        
        for i in range(aux_layers):
            
            temp_k = aux_kv[i][0] @ aux_weights["k_proj"][i]
            temp_v = aux_kv[i][1] @ aux_weights["v_proj"][i]
            
            temp_k = temp_k @ base_weights["k_proj"][i]
            temp_v = temp_v @ base_weights["v_proj"][i]
            
            pred_kv += ((temp_k, temp_v), )
        
        timer.start("base")         
        base_output = mygenerate(
            model=base,
            tokenizer=base_tokenizer,
            inputs=base_request,
            max_length=args["max_length"],
            max_new_tokens=args["max_new_tokens"],
            output_hidden_states=True,
            output_attentions=True,
            repetition_penalty=args["repetition_penalty"],
            use_cache=True,
            use_predicted_key_value=args['use_predicted_key_value'],
            predicted_key_value=pred_kv,
        )
        # base_output = mygenerate_deprecated(
        #     model=base,
        #     tokenizer=base_tokenizer,
        #     inputs=base_request,
        #     max_length=args["max_length"],
        #     max_new_tokens=args["max_new_tokens"],
        #     output_hidden_states=True,
        #     output_attentions=True,
        #     # repetition_penalty=args["repetition_penalty"],
        #     use_cache=True,
        #     use_predicted_key_value=args['use_predicted_key_value'],
        #     predicted_key_value=pred_kv,
        # )
        timer.end("base")
        
        preds += [''.join(pred.strip()) for pred in base_output]
        refs += [''.join(ref.strip()) for ref in base_refs]
        timer.end("total")
        
        if len(preds) % (args["batch_size"] * 5) == 0 or len(preds) == args["batch_size"]:
            logging.info(f"base_output : {json.dumps({'prediction':base_output}, indent=4)}")
            logging.info(f"base_refs : {json.dumps({'reference':base_refs}, indent=4)}")
            
            temp_scores = rouge.get_scores(hyps=preds, refs=refs, avg=True)
            temp_results = {key: value['f'] for key, value in temp_scores.items()}
            temp_results['avg'] = sum(temp_results.values()) / len(temp_results)
            logging.info(f"current time cost\t :\t {timer.get_time(mode='avg',name='base')}/{timer.get_time(mode='avg',name='total')} seconds")
            # HACK(zihao):notice that the base time cost now didn't exclude the KV generation itself
            logging.info(f"current {args['mode']} Rouge-1: {temp_results['rouge-1']:>0.2f} Rouge-2: {temp_results['rouge-2']:>0.2f} Rouge-L: {temp_results['rouge-l']:>0.2f} average: {temp_results['avg']}\n")
    
    scores = rouge.get_scores(hyps=preds, refs=refs, avg=True)
    results = {key: value['f'] for key, value in scores.items()}
    results['avg'] = sum(results.values()) / len(results)
    
    logging.info(f"total time cost\t :\t {timer.get_time(mode='sum',name='base')}/{timer.get_time(mode='sum',name='total')} seconds")
    logging.info(f"{args['mode']} Rouge-1: {results['rouge-1']:>0.2f} Rouge-2: {results['rouge-2']:>0.2f} Rouge-L: {results['rouge-l']:>0.2f} average: {results['avg']}\n")
    

if __name__ == "__main__":
    args = {
        "aux_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_name": "meta-llama/Llama-2-7b-hf",
        "dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        "batch_size": 1,
        "shuffle": True,
        "seed": 42,
        "loss_func": "all",
        "mode": "validation",
        "device": "cpu",
        "max_length": None,
        "max_new_tokens": 5,
        "use_prompt": True,
        "repetition_penalty": 1,
        "use_predicted_key_value": False,
        "use_pad": False,
        "tokenizer_padding": True,
        "tokenizer_truncation": True,
        "tokenizer_padding_side": "left",
    }
    if args["loss_func"] == "all":
        args["loss_func"] = SURPORTED_LOSS_FUNC
    logging.info(f"args : \n{json.dumps(args, indent=4)}")
    main(args)