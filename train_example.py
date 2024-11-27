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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from packges.utils import Timers
from packges.predictor import KVPredictor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from packges.utils import constants
from packges.losses import CustomLoss
from rouge import Rouge
# WandB – Import the wandb library
import wandb

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
handler = RotatingFileHandler(f'/workspace/log/examples/train_examples_{now}.log', maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        handler,  # output to file
        logging.StreamHandler()  # output to console
    ]
)

def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    aux: AutoModelForCausalLM,
    base: AutoModelForCausalLM,
    aux_tokenizer: AutoTokenizer,
    base_tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
):
    model.train()
    losses = constants()
    losses.register("loss")
    
    timer = Timers("request")
    timer.create_timer("total")
    timer.create_timer("train")
    timer.create_timer("prepare")
    
    for idx, batched_data in enumerate(dataloader, 0):
        timer.start("total")
        timer.start("prepare")
        aux_request = aux_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=args['max_lenth'])
        base_request = base_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=args['max_lenth'])
        
        aux_output = aux(**aux_request, output_attentions=True ,use_cache=True)
        base_output = base(**base_request, output_attentions=True ,use_cache=True)
        timer.end("prepare")
        
        aux_kv = aux_output.past_key_values
        base_kv = base_output.past_key_values
        
        logging.debug(f"aux_kv shape: {aux_kv[0][0].shape}")     # [batch_size, num_heads, num_tokens, head_dim]
        logging.debug(f"base_kv shape: {base_kv[0][0].shape}")   # to [batch_size, num_tokens, num_heads * head_dim]
        
        # training
        timer.start("train")
        optimizer.zero_grad()  # 清除梯度
        ins = [(k.to(args['device']), v.to(args['device'])) for k, v in aux_kv]
        pred_kv = model(ins)  # 前向传播
        
        logging.debug(f"pred_kv shape: {pred_kv[0][0].shape}")

        pred_kv = [(k.to(args['device']), v.to(args['device'])) for k, v in pred_kv]
        base_kv = [(k.to(args['device']), v.to(args['device'])) for k, v in base_kv]
        loss = loss_fn(pred_kv, base_kv)  # 计算损失
        
        logging.debug(f"loss : {loss}")
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        timer.end("train")
        timer.end("total")
        losses.update("loss",loss.item())
        
        wandb.log({"Training Loss": loss.item()})
        wandb.log({"Training total time": timer.get_time('total')})
        wandb.log({"Training train time": timer.get_time('train')})
        wandb.log({"Training prepare time": timer.get_time('prepare')})
        
        if idx % 400 == 0 or idx == len(dataloader) - 1 :
            logging.info(f"train batch {idx} epoch {epoch}\n"
                         f"---time cost:---\n"
                         f"total: {timer.get_time('total'):.2f}s, train: {timer.get_time('train'):.2f}s, prepare: {timer.get_time('prepare'):.2f}s\n"
                         f"---loss---\n"
                         f"loss: {loss.item()}\n"
                         )
    
    logging.info(
        f"{epoch} training epochs finished\n"
        f"total time cost\t :\t {timer.get_time('total', mode='sum'):.2f}\n"
        f"train time cost\t :\t {timer.get_time('train', mode='sum'):.2f}\n"
        f"prepare time cost\t :\t {timer.get_time('prepare', mode='sum'):.2f}\n"
        f"losses\t :\n{losses.report('loss', mode='last')}\n"
        )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'/workspace/results/ckpt/predictor_{epoch}_{now}.pth')

def eval_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    aux: AutoModelForCausalLM,
    base: AutoModelForCausalLM,
    aux_tokenizer: AutoTokenizer,
    base_tokenizer: AutoTokenizer,
    dataloader: DataLoader,
):
    model.eval()
    rouge = Rouge()
    losses = constants()
    losses.register("rouge-1")
    losses.register("rouge-2")
    losses.register("rouge-l")
    
    timer = Timers("request")
    timer.create_timer("total")
    timer.create_timer("eval")
    timer.create_timer("prepare")
    
    preds, refs = [], []
    
    for idx, batched_data in enumerate(dataloader, 0):
        timer.start("total")
        timer.start("prepare")
        aux_request = aux_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=args['max_lenth']).to(args['device'])
        # NOTICE(zihao): max_length + 1 for base model to reuse the past_key_values
        base_request = base_tokenizer(batched_data["question"], return_tensors="pt", padding='max_length', truncation=True, max_length=args['max_lenth'] + 1).to(args['device'])
        
        aux_output = aux(**aux_request, use_cache=True)
        timer.end("prepare")
        
        aux_kv = aux_output.past_key_values

        # evalidation
        timer.start("eval")
        ins = [(k.to(args['device']), v.to(args['device'])) for k, v in aux_kv]
        with torch.no_grad():
            pred_kv = model(ins)
        
        base_output = base.generate(
            **base_request,
            # output_attentions=True,
            max_new_tokens=8,
            do_sample=True,
            temperature=0.7,
            top_p=0.9, 
            top_k=8, 
            # num_beams=4,  # beam search:会导致hidden_states的shape不一致
            # early_stopping=True,
            # no_repeat_ngram_size=2,    # 禁止2-gram重复
            # past_key_values=pred_kv,
            use_cache=True,
            )
        
        # Truncate the input question from base_output
        base_output = base_output[:, base_request['input_ids'].shape[1]:]
        
        decoded_preds = base_tokenizer.batch_decode(base_output, skip_special_tokens=True)
        
        preds += [''.join(pred.strip()) for pred in decoded_preds]
        refs += [''.join(ref.strip()) for ref in batched_data["answer"]]
        
        print(f"preds : {preds}")
        print(f"refs : {refs}")
        
        scores = rouge.get_scores(hyps=preds, refs=refs, avg=True)
        result = {key: value['f'] * 100 for key, value in scores.items()}
        result['avg'] = np.mean(list(result.values()))
        
        logging.debug(f"Rouge-1: {result['rouge-1']:>0.2f} Rouge-2: {result['rouge-2']:>0.2f} Rouge-L: {result['rouge-l']:>0.2f}\n")
        timer.end("eval")      
        timer.end("total")
        losses.update("rouge-1",result['rouge-1'])
        losses.update("rouge-2",result['rouge-2'])
        losses.update("rouge-l",result['rouge-l'])
        
        wandb.log({"time": timer.get_time('total')})
        wandb.log({"Rouge-1": result['rouge-1']})
        wandb.log({"Rouge-2": result['rouge-2']})
        wandb.log({"Rouge-L": result['rouge-l']})
        
        if idx % 20 == 0 or idx == len(dataloader) - 1 :
            logging.info(f"eval batch {idx} epoch {epoch}\n"
                         f"---time cost:---\n"
                         f"total: {timer.get_time('total'):.2f}s, train: {timer.get_time('eval'):.2f}s, prepare: {timer.get_time('prepare'):.2f}s\n"
                         f"---loss---\n"
                         f"rouge-1: {losses.report('rouge-1')}\n"
                         f"rouge-2: {losses.report('rouge-2')}\n"
                         f"rouge-l: {losses.report('rouge-l')}\n"
                         )
    
    logging.info(
        f"{epoch} eval epochs finished\n"
        f"total time cost\t :\t {timer.get_time('total', mode='sum'):.2f}\n"
        f"train time cost\t :\t {timer.get_time('eval', mode='sum'):.2f}\n"
        f"prepare time cost\t :\t {timer.get_time('prepare', mode='sum'):.2f}\n"
        f"rouge-1 loss: {losses.report('rouge-1')}\n"
        f"rouge-2 loss: {losses.report('rouge-2')}\n"
        f"rouge-l loss: {losses.report('rouge-l')}\n"
        )

def main(
    args: Union[None, Optional[dict]] = None
):
    assert args is not None, "args is None"
    wandb.init(project=args['wandb_project'])
    
    aux_tokenizer = AutoTokenizer.from_pretrained(args["aux_name"])
    base_tokenizer = AutoTokenizer.from_pretrained(args["base_name"])
    aux_tokenizer.pad_token = aux_tokenizer.eos_token
    base_tokenizer.pad_token = base_tokenizer.eos_token
    aux_tokenizer.padding_side = "left"
    base_tokenizer.padding_side = "left"
    
    aux = AutoModelForCausalLM.from_pretrained(args["aux_name"], device_map="balanced_low_0", torch_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args["base_name"], device_map="balanced_low_0", torch_dtype=torch.bfloat16)
    
    train_dataset = loading_dataset(args["train_dataset"])
    eval_dataset = loading_dataset(args["eval_dataset"])
    train_dataloader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=args["shuffle"], collate_fn=collote_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args["eval_batch_size"], shuffle=args["shuffle"], collate_fn=collote_fn)
    
    logging.debug(f"aux.config : {aux.config}")
    logging.debug(f"base.config : {base.config}")
    
    aux_layers = aux.config.num_hidden_layers
    base_layers = base.config.num_hidden_layers

    model = KVPredictor(aux_layers, base_layers, aux.config.num_attention_heads, base.config.num_attention_heads, aux.config.hidden_size // aux.config.num_attention_heads, base.config.hidden_size // base.config.num_attention_heads)
    model.to(args['device']).bfloat16()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    if os.path.isfile(args["resume"]):
        checkpoint = torch.load(args["resume"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"resume from {args['resume']}")
    loss_fn = CustomLoss(base_layers)
    
    for epoch in range(args["epochs"]):
        # logging.info("Start training")
        # train_one_epoch(epoch, model, aux, base, aux_tokenizer, base_tokenizer, train_dataloader, optimizer, loss_fn)
        logging.info("Start evaluating")
        eval_one_epoch(epoch, model, aux, base, aux_tokenizer, base_tokenizer, eval_dataloader)
        scheduler.step()


if __name__ == "__main__":
    args = {
        "wandb_project": "kv-predictor",
        "aux_name": "facebook/opt-125m",
        "base_name": "facebook/opt-2.7b",
        "train_dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_train.json",
        "eval_dataset": "/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        "epochs":20,
        'max_lenth': 64,
        "train_batch_size": 80,
        "eval_batch_size": 32,
        "shuffle": True,
        "seed": 42,
        "resume": "/workspace/results/ckpt/predictor_0_2024-11-26-09-17-05.pth",
        "device": "cuda:0",
        "loss_func": "MSELoss",
    }

    logging.info(f"args : \n{args}")
    main(args)