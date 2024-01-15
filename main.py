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
from init import init_lora_model
from emtune import EM_finetune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='llama2')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset', type=str, default='./dataset/piqa')
    parser.add_argument('--lora_groupby', type=int, default=8)
    parser.add_argument('--lora_size', type=int, default=8)
    parser.add_argument('--em_epoch', type=int, default=20)
    parser.add_argument('--signal_train_epoch', type=int, default=3)
    parser.add_argument('--work_part', type=str, default='initial', choices=['initial', 'train', 'run'])

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def prepare_model():
    args = parse_args()
    setup_seed(args.seed)
    tonkenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.lora_size = args.lora_size
    config.lora_groupby = args.lora_groupby
    lora_model = redoLlmamaForCausalLM(config)
    if args.work_part == 'initial':
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        init_lora_model(model, lora_model,config).init()
        lora_model.save_pretrained(args.output_dir+'/init_model')
    elif args.work_part == 'train':
        model = LlamaForCausalLM.from_pretrained(args.output_dir+'/init_model')
        dataset = Dataset.load_dataset('json', data_files=args.dataset)
        em_finetune = EM_finetune(lora_model, config, dataset, args.em_epoch, args.signal_train_epoch, args.lora_groupby, args.output_dir)
        em_finetune.finetune()
        lora_model.save_pretrained(args.output_dir+'/finetune_model')
    elif args.work_part == 'run':
        pass
    
if __name__ == '__main__':
    prepare_model()
    

