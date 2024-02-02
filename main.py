from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaConfig
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
from configuration_llama import LlamaConfig as redoLlamaConfig
from init import initialize
from emtune import EM_finetune
from utils import *
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dataset_utils import evaluate as dataset_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='./../Llama-2-7b-hf')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset', type=str, default='./dataset/piqa')
    parser.add_argument('--lora_groupby', type=int, default=8)
    parser.add_argument('--lora_size', type=int, default=1024)
    parser.add_argument('--em_epoch', type=int, default=20)
    parser.add_argument('--use_attn_match', type=bool, default=False)
    parser.add_argument('--use_ffn_match', type=bool, default=False)
    parser.add_argument('--signal_train_epoch', type=int, default=3)
    parser.add_argument('--work_part', type=str, default='run', choices=['initial', 'train', 'run'])
    return parser.parse_args()

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def prepare_model():
    args = parse_args()
    setup_seed(args.seed)
    
    tonkenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = redoLlamaConfig()
    config.lora_size = args.lora_size
    config.lora_groupby = args.lora_groupby
    config.use_attn_match = args.use_attn_match
    config.use_ffn_match = args.use_ffn_match
    lora_model = redoLlmamaForCausalLM(config)
    
    if args.work_part == 'initial':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        lora_model.load_state_dict(model.state_dict(), strict=False)
        print("real model: ", model.model.norm.weight)
        print("lora model: ", lora_model.model.norm.weight)
        initialize(model, lora_model,config).init()
        # lora_model.save_pretrained(args.output_dir+'/init_model')
        torch.save(lora_model.state_dict(), args.output_dir+f'/init_state_{args.lora_groupby}_{args.lora_size}')
        lora_model = lora_model.to(device)
        instruction = ["hello,world("]
        print(evaluate(lora_model, tonkenizer, instruction))
        
    elif args.work_part == 'train':
        model = redoLlmamaForCausalLM.from_pretrained(args.output_dir+'/init_model')
        dataset = Dataset.load_dataset('json', data_files=args.dataset)
        em_finetune = EM_finetune(lora_model, config, dataset, args.em_epoch, args.signal_train_epoch, args.lora_groupby, args.output_dir)
        em_finetune.finetune()
        lora_model.save_pretrained(args.output_dir+'/finetune_model')
        
    elif args.work_part == 'run':
        # with init_empty_weights():
        #     model = redoLlmamaForCausalLM(config)
        # model = load_checkpoint_and_dispatch(
        #     model, args.output_dir+'/init_model', device_map="auto"
        # )
        model = redoLlmamaForCausalLM(config)
        model.load_state_dict(torch.load(args.output_dir+'/init_state_8_1024'),strict=False)
        model = model.to(device)
        dataset_evaluate(model, tonkenizer, 'piqa', device)
        
    
if __name__ == '__main__':
    prepare_model()
    

