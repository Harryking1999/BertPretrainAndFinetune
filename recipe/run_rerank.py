import os
import time

import torch.distributed
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import datetime
#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))#多卡时，数据量非常大的时候用
from src.prepare_data import prepare_train_dataset
from src.util.exp import setup
from src.model import BertForRerankPointwise
from src.util.trainer import RerankTrainerMultiFieldWithPointwise
from transformers import AutoTokenizer, AutoConfig

# import argparse

def main():
    exp_cfg, accelerator = setup(cfg_file="./config/retrieve.yaml")
    if not exp_cfg.get("tokenizer", None):
        exp_cfg.tokenizer = exp_cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.tokenizer)
    model_cfg = AutoConfig.from_pretrained(exp_cfg.model_name, output_hidden_states = True)
    model_cfg.update(
        {
            "rdrop": exp_cfg.get("rdrop", 0.5),
            "pointwise_weight": exp_cfg.get("pointwise_weight", 0.7),
            "pointwise_threshold": exp_cfg.get("pointwise_threshold", 0.0025),
            "classifier_dropout": exp_cfg.get("classifier_dropout", 0),
            "contrast_temperature": exp_cfg.get("contrast_temperature", 1),
        }
    )

    model = BertForRerankPointwise.from_pretrained(
        exp_cfg.model_name, config=model_cfg, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    # print(type(model))

    train_dataset = prepare_train_dataset(
        exp_cfg.train_data_file,
        accelerator,
        tokenizer,
        max_length = exp_cfg.max_length,
        num_proc = 64
    )
    test_dataset = prepare_train_dataset(
        exp_cfg.test_data_file,
        accelerator,
        tokenizer,
        max_length = exp_cfg.max_length,
        num_proc = 64
    )
    trainer = RerankTrainerMultiFieldWithPointwise(
        config = exp_cfg,
        accelerator = accelerator,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        test_dataset = test_dataset
    )
    trainer.train()

if __name__ == "__main__":
    main()