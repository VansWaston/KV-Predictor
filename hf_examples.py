# Description: Examples of using Hugging Face's Transformers library

# Importing necessary libraries
import torch
# datasets
# from packges.MyDatasets import datasets
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

# metrics
from evaluate import load

# model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# loading datasets
# step 1: load dataset
## method 1: load raw dataset from local file and transform dict-like dataset to hf-like dataset
# from packges.MyDatasets import loading_dataset
# prefered method: easier to modify and more flexible, but without under-layer optimization
# dataset = Dataset.from_dict(loading_dataset("/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json",use_prompt=True))

## method 2: directly load hf-like dataset from local file
dataset = load_dataset("json", data_files="/datasets/mandarjoshi/trivia_qa/rc.nocontext/rc_nocontext_validation.json")  # load hf-like dataset directly from local file

## step 2: prepare dataset(add prompt)
def preprend(example):
    return {"question":["question: "+ x for x in example["question"]]}
encoded_dataset = dataset.map(preprend, batched=True, batch_size=64)
print(encoded_dataset)

## step 3: split dataset
# train_test_dataset = encoded_dataset.train_test_split(test_size=0.1)
# train_dataset = train_test_dataset["train"]
# test_dataset = train_test_dataset["test"]

## step 4: prepare CustomDataset and DataLoader
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.question = self.data["question"]
        self.answer = self.data["answer"]

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        question = self.question[index]
        answer = self.answer[index]

        source = self.tokenizer.batch_encode_plus([question], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([answer], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# loading metrics
rouge_metric = load("rouge")

# loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

model = AutoModelForSeq2SeqLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

