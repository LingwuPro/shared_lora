from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from loraLlama import LlamaForCausalLM as redoLlamaForCausalLM
from datasets import Dataset
import torch.nn as nn
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
from collections import defaultdict
from typing import Dict, List, DefaultDict
from utils import get_attn_dict, reshape_ffn_weight
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class initialize:
    def __init__(self,
        model:LlamaForCausalLM, 
        lora_model: redoLlamaForCausalLM,
        config: LlamaConfig,
        ) -> None:
        
        self.model = model
        self.lora_size = config.lora_size
        self.lora_groupby = config.lora_groupby
        self.device = device
        self.config = config
        self.attn_list = config.attn_list
        self.num_layers = config.num_hidden_layers
        self.lora_model = lora_model
        self.use_attn_match = config.use_attn_match
        
        self.q_layer_dict: Dict[str, torch.Tensor] = {}
        self.k_layer_dict: Dict[str, torch.Tensor] = {}
        self.v_layer_dict: Dict[str, torch.Tensor] = {}
        self.o_layer_dict: Dict[str, torch.Tensor] = {}
        self.gate_layer_dict: Dict[str, torch.Tensor] = {}
        self.up_layer_dict: Dict[str, torch.Tensor] = {}
        self.down_layer_dict: Dict[str, torch.Tensor] = {}
        # for name, module in lora_model.model.named_modules():
        #     print(name)
        
                    
    def replace_state(self, config:LlamaConfig)->None:
        for name, module in self.model.named_modules():
            if not isinstance(module, Linear):
                continue
            desired_part = name.split('.')[0:3]
            name_string = '.'.join(desired_part) # model.layers.31
            if name.endswith("q_proj"):
                self.q_layer_dict[name_string] = module.weight
            elif name.endswith("k_proj"):
                self.k_layer_dict[name_string] = module.weight
            elif name.endswith("v_proj"):
                self.v_layer_dict[name_string] = module.weight
            elif name.endswith("o_proj"):
                self.o_layer_dict[name_string] = module.weight
            elif name.endswith("gate_proj"):
                self.gate_layer_dict[name_string] = module.weight
            elif name.endswith("up_proj"):
                self.up_layer_dict[name_string] = module.weight
            elif name.endswith("down_proj"):
                self.down_layer_dict[name_string] = module.weight
                
        if config.use_attn_match == True:
            attn_list = [str(i) for i in config.num_hidden_layers]
            self.attn_list = get_attn_dict(attn_list, self.lora_groupby, self.q_layer_dict, self.k_layer_dict, self.v_layer_dict, self.o_layer_dict)
            config.attn_list = self.attn_list
        else:
            group_size = self.num_layers // self.lora_groupby
            self.attn_list = []
            for idx in range(self.lora_groupby):
                temp_list = [str(i) for i in range(idx * group_size, (idx + 1) * group_size)]
                self.attn_list.append(temp_list)
            config.attn_list = self.attn_list
            for group in range(self.lora_groupby):
                for layer in range(group_size):
                    real_layer = group * group_size + layer
                    self.lora_model.model.layers[group].sub_layers[layer].self_attn.rotary_emb = self.model.model.layers[real_layer].self_attn.rotary_emb
                    self.lora_model.model.layers[group].sub_layers[layer].input_layernorm = self.model.model.layers[real_layer].input_layernorm
                    self.lora_model.model.layers[group].sub_layers[layer].post_attention_layernorm = self.model.model.layers[real_layer].post_attention_layernorm
                    self.lora_model.model.layers[group].sub_layers[layer].mlp.act_fn = self.model.model.layers[real_layer].mlp.act_fn
                    
        
        if config.use_ffn_match == True:
            print(self.attn_list)
            self.gate_layer_dict, self.up_layer_dict, self.down_layer_dict = reshape_ffn_weight(self.attn_list, self.gate_layer_dict, self.up_layer_dict, self.down_layer_dict)
            
 
    def avg_W(self) -> None:
        groupby_layer_size = self.num_layers // self.lora_groupby
        for groupby_index in range(self.lora_groupby):
            
            q_num_rows, q_num_cols = next(iter(self.q_layer_dict.values())).shape
            k_num_rows, k_num_cols = next(iter(self.k_layer_dict.values())).shape
            v_num_rows, v_num_cols = next(iter(self.v_layer_dict.values())).shape
            o_num_rows, o_num_cols = next(iter(self.o_layer_dict.values())).shape
            gate_num_rows, gate_num_cols = next(iter(self.gate_layer_dict.values())).shape
            up_num_rows, up_num_cols = next(iter(self.up_layer_dict.values())).shape
            down_num_rows, down_num_cols = next(iter(self.down_layer_dict.values())).shape
            
            q_weight = []
            k_weight = []
            v_weight = []
            o_weight = []
            gate_weight = []
            up_weight = []
            down_weight = []

            for layer_index in range(groupby_layer_size):
                if self.use_attn_match == True:
                    real_layer = self.attn_list[groupby_index][layer_index]
                else:
                    real_layer = groupby_index * groupby_layer_size + layer_index
                
                #W_q
                q_weight.append(self.q_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_k
                k_weight.append(self.k_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_v
                v_weight.append(self.v_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_o
                o_weight.append(self.o_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_gate
                gate_weight.append(self.gate_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_up
                up_weight.append(self.up_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                #W_down
                down_weight.append(self.down_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)].view(-1))
                
            if self.use_attn_match == True:
                self.lora_model.model.groupby_index.W_q.weight.copy_(torch.mean(torch.stack(q_weight), dim=0).view(q_num_rows, q_num_cols))
                self.lora_model.model.groupby_index.W_k.weight.copy_(torch.mean(torch.stack(k_weight), dim=0).view(k_num_rows, k_num_cols))
                self.lora_model.model.groupby_index.W_v.weight.copy_(torch.mean(torch.stack(v_weight), dim=0).view(v_num_rows, v_num_cols))
                self.lora_model.model.groupby_index.W_o.weight.copy_(torch.mean(torch.stack(o_weight), dim=0).view(o_num_rows, o_num_cols))
                self.lora_model.model.groupby_index.W_gate.weight.copy_(torch.mean(torch.stack(gate_weight), dim=0).view(gate_num_rows, gate_num_cols))
                self.lora_model.model.groupby_index.W_up.weight.copy_(torch.mean(torch.stack(up_weight), dim=0).view(up_num_rows, up_num_cols))
                self.lora_model.model.groupby_index.W_down.weight.copy_(torch.mean(torch.stack(down_weight), dim=0).view(down_num_rows, down_num_cols))
            else:
                self.lora_model.model.layers[groupby_index].W_q.weight = nn.Parameter(
                    torch.mean(torch.stack(q_weight), dim=0).view(q_num_rows, q_num_cols))
                self.lora_model.model.layers[groupby_index].W_k.weight = nn.Parameter(
                    torch.mean(torch.stack(k_weight), dim=0).view(k_num_rows, k_num_cols))
                self.lora_model.model.layers[groupby_index].W_v.weight = nn.Parameter(
                    torch.mean(torch.stack(v_weight), dim=0).view(v_num_rows, v_num_cols))
                self.lora_model.model.layers[groupby_index].W_o.weight = nn.Parameter(
                    torch.mean(torch.stack(o_weight), dim=0).view(o_num_rows, o_num_cols))
                self.lora_model.model.layers[groupby_index].W_gate.weight = nn.Parameter(
                    torch.mean(torch.stack(gate_weight), dim=0).view(gate_num_rows, gate_num_cols))
                self.lora_model.model.layers[groupby_index].W_up.weight = nn.Parameter(
                    torch.mean(torch.stack(up_weight), dim=0).view(up_num_rows, up_num_cols))
                self.lora_model.model.layers[groupby_index].W_down.weight = nn.Parameter(
                    torch.mean(torch.stack(down_weight), dim=0).view(down_num_rows, down_num_cols))

    def get_diff_svd(self) -> None:
        groupby_layer_size = self.num_layers // self.lora_groupby
        for groupby_index in tqdm(range(self.lora_groupby), desc="svd_groupby"):
            for layer_index in tqdm(range(groupby_layer_size), desc=f"layer {groupby_index}"):
                if self.use_attn_match == True:
                    real_layer = self.attn_list[groupby_index][layer_index]
                else:
                    real_layer = groupby_index * groupby_layer_size + layer_index
                if self.use_attn_match == True:
                    # W_q
                    q_diff = self.q_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_q.weight
                    U, sigma, V = torch.svd(q_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.self_attn.A_q.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.self_attn.B_q.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_k
                    k_diff = self.k_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_k.weight
                    U, sigma, V = torch.svd(k_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.self_attn.A_k.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.self_attn.B_k.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_v
                    v_diff = self.v_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_v.weight
                    U, sigma, V = torch.svd(v_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.self_attn.A_v.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.self_attn.B_v.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_o
                    o_diff = self.o_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_o.weight
                    U, sigma, V = torch.svd(o_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.self_attn.A_o.weight.copy_(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.self_attn.B_o.weight.copy_(torch.sqrt(sigma) @ V)
                    # W_gate
                    gate_diff = self.gate_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_gate.weight
                    U, sigma, V = torch.svd(gate_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.mlp.A_gate.weight.copy_(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.mlp.B_gate.weight.copy_(torch.sqrt(sigma) @ V)
                    # W_up
                    up_diff = self.up_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_up.weight
                    U, sigma, V = torch.svd(up_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.mlp.A_up.weight.copy_(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.mlp.B_up.weight.copy_(torch.sqrt(sigma) @ V)
                    # W_down
                    down_diff = self.down_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.groupby_index.W_down.weight
                    U, sigma, V = torch.svd(down_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers.real_layer.mlp.A_down.weight.copy_(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers.real_layer.mlp.B_down.weight.copy_(torch.sqrt(sigma) @ V)
                else:
                    # W_q
                    q_diff = self.q_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_q.weight
                    U, sigma, V = torch.svd(q_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    print(q_diff)
                    # temp = U @ torch.sqrt(sigma)
                    # print(temp.shape, self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.A_q.weight.shape)
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.A_q.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.B_q.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_k
                    k_diff = self.k_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_k.weight
                    U, sigma, V = torch.svd(k_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.A_k.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.B_k.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_v
                    v_diff = self.v_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_v.weight
                    U, sigma, V = torch.svd(v_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.A_v.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.B_v.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_o
                    o_diff = self.o_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_o.weight
                    U, sigma, V = torch.svd(o_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.A_o.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].self_attn.B_o.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_gate
                    gate_diff = self.gate_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_gate.weight
                    U, sigma, V = torch.svd(gate_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.A_gate.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.B_gate.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_up
                    up_diff = self.up_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_up.weight
                    U, sigma, V = torch.svd(up_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.A_up.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.B_up.weight = nn.Parameter(torch.sqrt(sigma) @ V)
                    # W_down
                    down_diff = self.down_layer_dict["model.layers.{real_layer}".format(real_layer = real_layer)] - self.lora_model.model.layers[groupby_index].W_down.weight
                    U, sigma, V = torch.svd(down_diff)
                    U = U[:,:self.lora_size]
                    sigma = torch.diag(sigma[:self.lora_size])
                    V = V[:,:self.lora_size].T
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.A_down.weight = nn.Parameter(U @ torch.sqrt(sigma))
                    self.lora_model.model.layers[groupby_index].sub_layers[layer_index].mlp.B_down.weight = nn.Parameter(torch.sqrt(sigma) @ V)
    
    def init(self)->None:
        self.replace_state(self.config)
        self.avg_W()
        self.get_diff_svd()            
