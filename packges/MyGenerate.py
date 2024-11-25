from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from transformers.generation.configuration_utils import GenerationConfig
import json
import logging

# TODO(zihao): not good as model.generate because of the logit_process and other stuffs
def mygenerate_deprecated(
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    inputs ,
    max_length : int = 20,
    max_new_tokens : int = 20,
    output_hidden_states : bool = False,
    output_attentions : bool = False,
    use_cache : bool = True,
    predicted_key_value : Optional[Tuple[Tuple[torch.Tensor]]] = None,
    use_predicted_key_value : bool = False,
):
    pred_ids = []
    eos_token_id = tokenizer.eos_token_id
    
    is_done = torch.full((inputs["input_ids"].shape[0],), False, dtype=torch.bool)
    for i in range(max_new_tokens):
        outputs = model(
            **inputs, 
            output_hidden_states=output_hidden_states, 
            output_attentions=output_attentions, 
            use_cache=use_cache, 
            predicted_key_value=predicted_key_value if i == 0 and use_predicted_key_value else None, 
        )
        logits = outputs.logits[:, -1, :]   # Greedy decoding
        pred_id = torch.argmax(logits, dim=-1)
        
        is_done = is_done | (pred_id == eos_token_id)
        if all(is_done):
            break
        
        if i == 0:
            pred_ids = pred_id.unsqueeze(1)
        else:
            pred_ids = torch.cat([pred_ids, pred_id.unsqueeze(1)], dim=-1)        
        # update the inputs
        inputs["input_ids"] = torch.cat([inputs["input_ids"], pred_id.unsqueeze(1)], dim=-1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(pred_id).unsqueeze(1)], dim=-1)
        
        del outputs
    
    pred_ids = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    logging.debug(f"pred_ids : {pred_ids}")
    return pred_ids

def mygenerate(
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    inputs ,
    do_sample : bool = False,
    early_stopping : bool = False,
    num_beams : int = 1,
    temperature : float = 1.0,
    max_length : int = None,
    max_new_tokens : int = None,
    min_length : int = 0,
    output_hidden_states : bool = True,
    output_attentions : bool = True,
    output_scores : bool = True,
    use_cache : bool = True,
    length_penalty : float = 1.0,
    repetition_penalty : float = 1.0,
    return_dict_in_generate : bool = False,
    
    use_predicted_key_value : bool = False,
    predicted_key_value : Optional[Tuple[Tuple[torch.Tensor]]] = None,
):
    kwargs = {
        "do_sample": do_sample,
        "early_stopping": early_stopping,
        "num_beams": num_beams,
        "temperature": temperature,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "min_length": min_length,
        "output_hidden_states": output_hidden_states,
        "output_attentions": output_attentions,
        "output_scores": output_scores,
        "use_cache": use_cache,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "return_dict_in_generate": return_dict_in_generate,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    model_kwargs = {
        "predicted_key_value": predicted_key_value if use_predicted_key_value else None,
    }
    
    generationconfig = GenerationConfig(**kwargs)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generationconfig,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **model_kwargs,
    )
    
    pred_ids = outputs.sequences[:,inputs["input_ids"].shape[1]:] if return_dict_in_generate else outputs[:,inputs["input_ids"].shape[1]:]
    # logging.info(f"pred_ids : {json.dumps({'pred_ids':tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)}, indent=4)}")
    pred_ids = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return pred_ids
    
    