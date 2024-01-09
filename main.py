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

if __name__ == "__main__":

    device = "cuda"

    config = LlamaConfig()
    model = LlamaForCausalLM(config)
    model_new = redoLlmamaForCausalLM(config)
    for name, module in model_new.named_modules():
        print(name)

