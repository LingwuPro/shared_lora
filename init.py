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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class initialize:
    def __init__(self,
        model:LlamaForCausalLM, 
        lora_model: redoLlmamaForCausalLM,
        lora_size: int = 8,
        lora_groupby: int = 8,
        config: LlamaConfig,
        ) -> None:
        
        self.model = model
        self.lora_size = lora_size
        self.lora_groupby = lora_groupby
        self.device = device
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        self.q_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.k_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.v_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.o_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.gate_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.up_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.down_layer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        self.state_dict = defaultdict()
        
        self.sample_cfg = {
            "token_size": sample_token_size,
            "data_size": sample_data_size,
        }
        
        self.hook_catch = []
        
    def get_weight(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, Linear):
                continue
            desired_part = name.split('.')[0:3]
            name_string = '.'.join(desired_part) + '.' # model.layers.31
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
    
    def avg_W(self) -> None:
        groupby_layer_size = self.num_layers // self.lora_groupby
        for groupby_index in range(self.lora_groupby):
            
            q_num_rows, q_num_cols = next(iter(q_layer_dict.values())).shape
            k_num_rows, k_num_cols = next(iter(q_layer_dict.values())).shape
            v_num_rows, v_num_cols = next(iter(q_layer_dict.values())).shape
            o_num_rows, o_num_cols = next(iter(q_layer_dict.values())).shape
            gate_num_rows, gate_num_cols = next(iter(q_layer_dict.values())).shape
            up_num_rows, up_num_cols = next(iter(q_layer_dict.values())).shape
            down_num_rows, down_num_cols = next(iter(q_layer_dict.values())).shape
            
            for layer_index in range(groupby_layer_size):
                real_layer = groupby_index * groupby_layer_size + layer_index
                
                #W_q
                q_weight = []
                q_weight.append(q_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_k
                k_weight = []
                k_weight.append(k_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_v
                v_weight = []
                v_weight.append(v_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_o
                o_weight = []
                o_weight.append(o_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_gate
                gate_weight = []
                gate_weight.append(gate_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_up
                up_weight = []
                up_weight.append(up_layer_dict["model.layers.{real_layer}"].view(-1))
                #W_down
                down_weight = []
                down_weight.append(down_layer_dict["model.layers.{real_layer}"].view(-1))

            lora_model.layers.groupby_index.W_q.weight.copy_(
                torch.mean(torch.stack(q_weight), dim=0).view(q_num_rows, q_num_cols))
            lora_model.layers.groupby_index.W_k.weight.copy_(
                torch.mean(torch.stack(k_weight), dim=0).view(k_num_rows, k_num_cols))
            lora_model.layers.groupby_index.W_v.weight.copy_(
                torch.mean(torch.stack(v_weight), dim=0).view(v_num_rows, v_num_cols))
            lora_model.layers.groupby_index.W_o.weight.copy_(
                torch.mean(torch.stack(o_weight), dim=0).view(o_num_rows, o_num_cols))
            lora_model.layers.groupby_index.W_gate.weight.copy_(
                torch.mean(torch.stack(gate_weight), dim=0).view(gate_num_rows, gate_num_cols))
            lora_model.layers.groupby_index.W_up.weight.copy_(
                torch.mean(torch.stack(up_weight), dim=0).view(up_num_rows, up_num_cols))
            lora_model.layers.groupby_index.W_down.weight.copy_(
                torch.mean(torch.stack(down_weight), dim=0).view(down_num_rows, down_num_cols))

    def get_diff_svd(self) -> None:
        groupby_layer_size = self.num_layers // self.lora_groupby
        for groupby_index in range(self.lora_groupby):
            for layer_index in range(groupby_layer_size):
                real_layer = groupby_index * groupby_layer_size + layer_index
                # W_q
                q_diff = q_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_q.weight
                U, sigma, V = torch.svd(q_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_q.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_q.weight.copy_(torch.sqrt(sigma) @ V)
                # W_k
                k_diff = k_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_k.weight
                U, sigma, V = torch.svd(k_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_k.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_k.weight.copy_(torch.sqrt(sigma) @ V)
                # W_v
                v_diff = v_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_v.weight
                U, sigma, V = torch.svd(v_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_v.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_v.weight.copy_(torch.sqrt(sigma) @ V)
                # W_o
                o_diff = o_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_o.weight
                U, sigma, V = torch.svd(o_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_o.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_o.weight.copy_(torch.sqrt(sigma) @ V)
                # W_gate
                gate_diff = gate_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_gate.weight
                U, sigma, V = torch.svd(gate_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_gate.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_gate.weight.copy_(torch.sqrt(sigma) @ V)
                # W_up
                up_diff = up_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_up.weight
                U, sigma, V = torch.svd(up_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_up.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_up.weight.copy_(torch.sqrt(sigma) @ V)
                # W_down
                down_diff = down_layer_dict["model.layers.{real_layer}"] - lora_model.layers.groupby_index.W_down.weight
                U, sigma, V = torch.svd(down_diff)
                U = U[:,:self.lora_size]
                sigma = torch.diag(sigma[:self.lora_size])
                V = V[:,:self.lora_size].T
                model.layers.groupby_index.sub_layers.layer_index.self_attn.A_down.weight.copy_(U @ torch.sqrt(sigma))
                model.layers.groupby_index.sub_layers.layer_index.self_attn.B_down.weight.copy_(torch.sqrt(sigma) @ V)
    
    def init(self)->None:
        get_weight()
        avg_W()
        get_diff_svd()            
            
        
                
        
        
    
