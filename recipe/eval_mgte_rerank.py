import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from datasets import load_dataset

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import ndcg_score, roc_auc_score
from src.model import BertForRerankPointwise

device = torch.device('cuda')

model_name_or_path = "your model path"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model_cfg = AutoConfig.from_pretrained(
    model_name_or_path, num_labels=1, trust_remote_code=True
)
model = BertForRerankPointwise.from_pretrained(
    model_name_or_path, config=model_cfg, ignore_mismatched_sizes=True
).to(device)

input_file = "test file path"

merge_dataset = load_dataset("json", data_files=input_file, split="train")
def filter_constant_scores(example):
    scores = set([doc['sim_score'] for doc in example['doc']])
    return len(scores) > 1
merge_dataset = merge_dataset.filter(filter_constant_scores)

auc_scores = []
k_list = ['all', 1, 3, 5, 10]
ndcg_scores = {k: [] for k in k_list}
cros_auc_scores = deepcopy(auc_scores)
cros_ndcg_scores = deepcopy(ndcg_scores)

ref_ls = []
pred_ls = []
for i in tqdm(merge_dataset):
    ref = []
    doc_str = []
    for j in i['doc']:
        ref.append(j['sim_score'])
        doc_str.append(i['query']  + '</s>' + j['name'] + '</s>' + j['addr'] + '</s>')
    batch_dict = tokenizer(doc_str, max_length=256, padding=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    similarity = None
    with torch.no_grad():
        outputs = model(**batch_dict)
        pooler_outputs = outputs.logits.view(-1)

        pred = pooler_outputs.tolist()
        similarity = [pred]

    ref_ls.append(ref)
    pred_ls.append(similarity[0])
    for k in ndcg_scores.keys():
        if k == 'all':
            ndcg_scores[k].append(ndcg_score([ref], similarity))
        else:
            ndcg_scores[k].append(ndcg_score([ref], similarity, k=k))
    binary_threshold = 2
    binary_label = [1 if r >= binary_threshold else 0 for r in ref]

    if(len(set(binary_label)) > 1):
        auc_scores.append(roc_auc_score(binary_label, similarity[0]))

report = ""
report += f"\tauc:\t{np.asarray(auc_scores).mean()}\n"
for k in ndcg_scores.keys():
    score = np.asarray(ndcg_scores[k]).mean()
    if k == 'all':
        report += f"\tndcg:{score}\n"
    else:
        report += f"\tndcg@{k}:\t{score}\n"
logging.info(report)
