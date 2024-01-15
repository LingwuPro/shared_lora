from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from .loraLlama import LlamaForCausalLM as redoLlmamaForCausalLM
from datasets import Dataset
from transformers import DataCollator
import transformers
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
from utils import load_training_dataset

class EM_finetune:
    def __init__(self,
                 lora_model:redoLlmamaForCausalLM,
                 config:LlamaConfig,
                 dataset:Dataset,
                 em_epoch:int = 20,
                 signal_train_epoch:int = 3,
                 lora_groupby:int = 8,
                 output_dir:str = "./output"
                 ) -> None:
        self.lora_model = lora_model
        self.config = config
        self.dataset = dataset
        self.em_epoch = em_epoch
        self.signal_train_epoch = signal_train_epoch
        self.lora_groupby = lora_groupby
        self.output_dir = output_dir
        
    def finetune(self) -> None:
        train_data, val_data, tokenizer = load_training_dataset(self.dataset)
        trainer = transformers.Trainer(
            model=self.lora_model,
            train_dataset=train_data,
            eval_dataset=None,
            args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=32,
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=3e-4,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            eval_steps=None,
            save_strategy="steps",
            save_steps=30,
            output_dir=self.output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
            )
        )
        trainer.train()
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
