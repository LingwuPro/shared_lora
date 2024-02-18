from transformers import LlamaTokenizer, AutoTokenizer
# from loraLlama import LlamaForCausalLM as redoLlmamaForCausalLM
from modeling_llama import LlamaForCausalLM
from datasets import Dataset
from transformers import DataCollator, GenerationConfig
from torch.nn import Linear
import os
import sys
import random
import numpy as np
import torch
import logging
import argparse
from configuration_llama import LlamaConfig
from networkx import Graph
from typing import Dict, List, Tuple, DefaultDict, Set
from torch import Tensor
import networkx as nx
from tqdm import tqdm
import faiss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_layer_pattern(idx: int, attn_list: list) -> int:
    """
    Get the layer pattern for a given layer index.
    """
    if attn_list is None:
        ValueError("attn_list is None")
    if idx >= len(attn_list):
        ValueError(f"idx {idx} is out of range of attn_list {attn_list}")
    for i, layer_pattern in enumerate(attn_list):
        if idx in layer_pattern:
            return i
        
def get_singal_attn_edge_weight(mat1: torch.Tensor, mat2: torch.Tensor) -> float:
    """
    use mat1 and mat2 's Dot Product to calculate the edge weight
    """
    assert mat1.shape == mat2.shape
    result = torch.sum(torch.mul(mat1, mat2))
    return result.item()

def std_matching(matching: Set[Tuple[int, int]]) -> List[int]:
    """
    Standardize the matching result.
    """
    result = set()
    for pair in matching:
        if pair[0] < pair[1]:
            result.add(pair)
        else:
            result.add((pair[1], pair[0]))
    result = list(result)
    result = sorted(result, key=lambda x: x[1])
    for idx, pair in enumerate(result):
        result[idx] = pair[0]
    return result

def get_attn_dict(attn_dict: list[str], lora_groupby: int, 
                  q_dict:Dict[str, List[Tensor]], k_dict:Dict[str, List[Tensor]],
                  v_dict:Dict[str, List[Tensor]], o_dict:Dict[str, List[Tensor]],
                  ) -> List[List[int]]:
    num_layer = len(attn_dict)
    assert num_layer == len(q_dict) == len(k_dict) == len(v_dict) == len(o_dict)
    whole_circle_num = num_layer // lora_groupby
    now_circle_num = 1
    while now_circle_num <= whole_circle_num:
        assert now_circle_num % 2 == 0
        G = nx.Graph()
        num_layer = len(attn_dict)
        for idx1, weight_1 in enumerate(num_layer):
            for idx2, weight_2 in enumerate(num_layer):
                if idx1 <= idx2:
                    continue
                sum_weight = 0
                for i in attn_dict[weight_1].split(','):
                    for j in attn_dict[weight_2].split(','):
                        assert i == j
                        q_1 = q_dict['model.layers.{i}'.format(i=i)]
                        k_1 = k_dict['model.layers.{i}'.format(i=i)]
                        v_1 = v_dict['model.layers.{i}'.format(i=i)]
                        o_1 = o_dict['model.layers.{i}'.format(i=i)]
                        
                        q_2 = q_dict['model.layers.{i}'.format(i=i)]
                        k_2 = k_dict['model.layers.{i}'.format(i=i)]
                        v_2 = v_dict['model.layers.{i}'.format(i=i)]
                        o_2 = o_dict['model.layers.{i}'.format(i=i)]
                        
                        sum_weight += get_singal_attn_edge_weight(q_1, q_2)
                        sum_weight += get_singal_attn_edge_weight(k_1, k_2)
                        sum_weight += get_singal_attn_edge_weight(v_1, v_2)
                        sum_weight += get_singal_attn_edge_weight(o_1, o_2)
                        
                G.add_edge(weight_1, weight_2, weight=sum_weight)  
        matching = nx.max_weight_matching(G, maxcardinality=True)
        attn_dict = []
        for part1, part2 in list(matching):
            attn_dict.append(part1 + ',' + part2)
        now_circle_num *= 2
    
    result = [[int(num) for num in attn.split(',')] for attn in attn_dict]
    return result

def reshape_ffn_weight(attn_list: List[List[int]], 
                       gate_layer_dict: Dict[str, torch.Tensor], 
                       up_layer_dict: Dict[str, torch.Tensor], 
                       down_layer_dict: Dict[str, torch.Tensor],
                       ) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
    gate_shape = gate_layer_dict['model.layers.0'].shape
    up_shape = up_layer_dict['model.layers.0'].shape
    down_shape = down_layer_dict['model.layers.0'].shape
    assert gate_shape[0] == up_shape[0] == down_shape[1]
    with torch.no_grad():    
        for group in attn_list:
            for idx, layer in tqdm(enumerate(group),desc="group: "):
                if idx == 0:
                    continue
                
                aim_gate = gate_layer_dict['model.layers.{i}'.format(i=group[0])]
                aim_up = up_layer_dict['model.layers.{i}'.format(i=group[0])]
                aim_down = down_layer_dict['model.layers.{i}'.format(i=group[0])].T
                aim = torch.cat([aim_gate, aim_up, aim_down], dim = 1)
                index = faiss.IndexIDMap(faiss.IndexFlatL2(gate_shape[1]*3))
                index.add_with_ids(aim.numpy(), np.arange(len(aim)))
                    
                reshape_gate = gate_layer_dict['model.layers.{i}'.format(i=layer)]
                reshape_up = up_layer_dict['model.layers.{i}'.format(i=layer)]
                reshape_down = down_layer_dict['model.layers.{i}'.format(i=layer)].T
                reshape = torch.cat([reshape_gate, reshape_up, reshape_down], dim=1)
                order_list = []
                for i in tqdm(range(gate_shape[0]), desc=f'layer{idx}_get_match : '):
                    distances, indices = index.search(reshape[i:i+1, :].numpy(), 1)  
                    order_list.append(indices[0][0])
                    index.remove_ids(np.array([indices[0][0]]))
                # print(order_list)                
                order_list = torch.tensor(order_list)
                gate_layer_dict['model.layers.{i}'.format(i=layer)] = gate_layer_dict['model.layers.{i}'.format(i=layer)][order_list]
                up_layer_dict['model.layers.{i}'.format(i=layer)] = up_layer_dict['model.layers.{i}'.format(i=layer)][order_list]
                down_layer_dict['model.layers.{i}'.format(i=layer)] = down_layer_dict['model.layers.{i}'.format(i=layer)][:, order_list]
                print(down_layer_dict['model.layers.{i}'.format(i=layer)].shape)
    return gate_layer_dict, up_layer_dict, down_layer_dict

def evaluate(
    models,
    tokenizer,
    instruction,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=32,
    stream_output=False,
    **kwargs,
):
    inputs = tokenizer(instruction, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = models.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    return output

# attn_list = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23],[24,25,26,27],[28,29,30,31]]
# idx = 17
# print(get_layer_pattern(idx, attn_list))
# matrix1 = torch.tensor([[1, 2], [3, 4]])
# matrix2 = torch.tensor([[5, 6], [7, 8]])
# print(get_attn_edge_weight(matrix1, matrix2))
