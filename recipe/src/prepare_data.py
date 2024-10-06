import json
import os
import random
import re
from glob import glob
from typing import List

import numpy as np
import torch as th
from accelerate import Accelerator
from datasets import (
    ClassLabel,
    Dataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk
)
from transformers import PreTrainedTokenizerBase

def prepare_train_dataset(
    data_file: str,
    accelerator: Accelerator = None,
    tokenizer: PreTrainedTokenizerBase = None,
    max_length: int = 128,
    num_proc: int = 8
):
    train_dataset = load_dataset("json", data_files=data_file, split="train")

    def tokenize_function(examples):
        tokenize_query = tokenizer(
            examples['query'],
            add_special_tokens = False,
            padding = False,
            truncation = True,
            max_length = max_length,
            is_split_into_words = False,
            return_token_type_ids = False,
            retrun_attention_mask = False,
        )['input_ids']

        pos_pointwise_label = []
        neg_pointwise_label = []
        pdoc_name, pdoc_addr = [], []
        for doc in examples['pdoc']:
            pdoc_name.append(doc['name'])
            pdoc_addr.append(doc['addr'])
            pos_pointwise_label.append(doc['sim_score'])
        ndoc_name, ndoc_addr = [], []
        for doc in examples['ndoc']:
            ndoc_name.append(doc['name'])
            ndoc_addr.append(doc['addr'])
            neg_pointwise_label.append(doc['sim_score'])
        tokenized_pdoc_name = tokenizer(
            pdoc_name,
            add_special_tokens = False,
            padding = False,
            truncation = True,
            max_length = max_length,
            is_split_into_words = False,
            return_token_type_ids = False,
            retrun_attention_mask = False
        )['input_ids']
        tokenized_pdoc_addr = tokenizer(
            pdoc_name,
            add_special_tokens = False,
            padding = False,
            truncation = True,
            max_length = max_length,
            is_split_into_words = False,
            return_token_type_ids = False,
            retrun_attention_mask = False
        )['input_ids']

        tokenized_ndoc_name = tokenizer(
            ndoc_name,
            add_special_tokens = False,
            padding = False,
            truncation = True,
            max_length = max_length,
            is_split_into_words = False,
            return_token_type_ids = False,
            retrun_attention_mask = False
        )['input_ids']
        tokenized_ndoc_addr = tokenizer(
            ndoc_name,
            add_special_tokens = False,
            padding = False,
            truncation = True,
            max_length = max_length,
            is_split_into_words = False,
            return_token_type_ids = False,
            retrun_attention_mask = False
        )['input_ids']
        
        tokenized_inputs = {}
        tokenized_inputs['tokenized_query'] = tokenize_query
        tokenized_inputs['tokenized_pdoc'] = list(
            zip(tokenized_pdoc_name, tokenized_pdoc_addr)
        )
        tokenized_inputs['tokenized_ndoc'] = list(
            zip(tokenized_ndoc_name, tokenized_ndoc_addr)
        )
        tokenized_inputs['pos_pointwise_label'] = pos_pointwise_label
        tokenized_inputs['neg_pointwise_label'] = neg_pointwise_label

        return tokenized_inputs
    with accelerator.main_process_first():
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched = False,
            num_proc = num_proc,
            remove_columns = train_dataset.column_names,
        )