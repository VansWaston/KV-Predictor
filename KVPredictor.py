from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
from datasets import batching
from vllm.model_executor.models.llama import LlamaAttention
from transformers import LlamaConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.config import CacheConfig
import torch
from vllm.attention import Attention, AttentionMetadata
import argparse

def load_dataset(file_path, return_type="Dict"):
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

class KVCached_LlamaAttn(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            cache_config=cache_config,
            prefix=prefix,
        )
        self.KV_cached = []
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        self.KV_cached.append([k, v])
        
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

class KVPredictor:
    pass

def main(args):
    # Load the dataset.
    dataset = load_dataset(args.dataset)
    
    # Create an LLM and tokenizer.
    aux = LLM(
        model=args.aux_model,
        gpu_memory_utilization=args.aux_gpu_memory_utilization,
        dtype=args.dtype,
    )
    # base = LLM(
    #     model=args.base_model,
    #     gpu_memory_utilization=args.base_gpu_memory_utilization,
    #     dtype=args.dtype,
    # )
    tokenizer = AutoTokenizer.from_pretrained(args.aux_model)
    aux.set_tokenizer(tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # base.set_tokenizer(tokenizer)
    
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Batch the dataset.
    if args.batch_size > 0:
        dataset = batching(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        
    # if not had_weights:
    aux_qkv_weights = aux_layers[0].self_attn.qkv_proj.weight
    print(f"aux_qkv_weights : {aux_qkv_weights}")
    
    for request in dataset:
        aux.generate(request, sampling_params)
        # base.generate(request, sampling_params)
        
        # Get the past key values from each layer of the aux.
        aux_layers = aux.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        aux_num_layer = len(aux_layers)
        
        # Get the KV weights of the aux model.
        
        
        # Get the past key values from each layer of the base.
        # base_layers = base.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        # base_num_layer = len(base_layers)
        
        # for j in range(base_num_layer):
        #     past_key_values = base_layers[j].self_attn.KV_cached
            
    
    pass

def parsing_args():
    parser = argparse.ArgumentParser(description="KV Predictor")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/zihao/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",
        help="Path to dataset",
    )
    parser.add_argument(
        "--aux_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to auxiliary model",
    )
    # parser.add_argument(
    #     "--base_model",
    #     type=str,
    #     help="Path to base model",
    # )
    parser.add_argument(
        "--aux_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization of the auxiliary model",
    )
    # parser.add_argument(
    #     "--base_gpu_memory_utilization",
    #     type=float,
    #     default=0.95,
    #     help="GPU memory utilization of the base model",
    # )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum tokens",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parsing_args()
    main(args=args)