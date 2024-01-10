from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from .loraLlama import LlamaForCausalLM as redoLlmamaForCausalLM
from datasets import Dataset
from transformers import DataCollator
from configuration_llama import LlamaConfig
from torch.nn import Linear
import os
import sys
import random
import numpy as np
import torch
import logging
import argparse
from tqdm import tqdm

class EM_finetune:
    def __init__(self,
                 lora_model:redoLlmamaForCausalLM,
                 config:LlamaConfig,
                 dataset:Dataset,
                 em_epoch:int = 20,
                 signal_train_epoch:int = 3,
                 lora_groupby:int = 8,
                 ) -> None:
        self.lora_model = lora_model
        self.config = config
        self.dataset = dataset
        self.em_epoch = em_epoch
        self.signal_train_epoch = signal_train_epoch
        self.lora_groupby = lora_groupby
        
    def finetune(self) -> None:
        pass
    
    def Expectation(self) -> None:
        for name, param in self.lora_model.named_parameters():
            param.requires_grad = False
            if name.startswith(r"model\.layers\.\d\.W_"):
                param.requires_grad = True
                print(name)
        self.finetune()
    
    def Maximum(self) -> None:
        for name, param in self.lora_model.name_parameters():
            param.requires_grad = False
            if "A_" in name or "B_" in name:
                param.requires_grad = True
                print(name)
        self.finetune()
        
    def train(self)->None:
        for epoch in tqdm(self.em_epoch):
            if epoch % 2 == 0:
                self.Expectation()
            else:
                self.Maximum()