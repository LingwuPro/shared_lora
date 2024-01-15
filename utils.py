from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from loraLlama import LlamaForCausalLM as redoLlmamaForCausalLM
from datasets import Dataset
from transformers import DataCollator
from torch.nn import Linear
import os
import sys
import random
import numpy as np
import torch
import logging
import argparse
from configuration_llama import LlamaConfig



def data_padding(data_point, tokenizer: LlamaTokenizer):
    pass

def load_training_dataset(dataset : Dataset) -> (Dataset, Dataset, LlamaTokenizer):
    train_valid_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    tokenizer = LlamaTokenizer.from_pretrained("...")
    train_data : Dataset = train_valid_split["train"].shuffle().map(
        data_padding(tokenizer)
    )
    dev_data : Dataset = train_valid_split["test"].shuffle().map(
        data_padding(tokenizer)
    )
    save_columns = ['input_ids', 'attention_mask', 'labels']
    train_data = train_data.remove_columns(
        set(train_data['train'].column_names) - set(save_columns))
    val_data = val_data.remove_columns(
        set(dev_data['train'].column_names) - set(save_columns))
    
    return train_data, val_data, tokenizer
