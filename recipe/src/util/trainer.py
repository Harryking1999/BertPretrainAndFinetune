import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch as th
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler
)

from .data_collator import (
    DataCollatorForTriplet,
    DataCollatorForRerankMultiFieldWithPointwiseLabel
)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1, emb_name='word_embeddings.'):
        # emb_name参数要换成模型中的embedding的参数名
        for name, param in self.model.named_parameters():
            if(param.requires_grad and emb_name in name):
                self.backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if(norm != 0 and not th.isnan(norm)):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if(param.requires_grad and emb_name in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
@dataclass
class BaseTrainer(ABC):
    config: DictConfig
    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __post_init__(self):
        self.prepare()
    
    @abstractmethod
    def prepare(self):
        raise NotImplementedError()
    
    def save_ckpt(
        self,
        unwrapped_model = None,
        completed_epochs: int = None,
        completed_steps: int = None,
    ):
        if(unwrapped_model == None):
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)

        if(
            not self.accelerator.is_main_process
            or self.config.get("exp_dir") == None
            or self.config.get("debug") == True
        ):
            return
        
        ckpt_dir = os.path_join(self.config.exp_dir, "checkpoint")
        if(completed_steps != None):
            ckpt_dir = os.path.join(ckpt_dir, f"step_{completed_steps}")
        elif(completed_epochs != None):
            ckpt_dir = os.path.join(ckpt_dir, f"epoch_{completed_epochs}")

        unwrapped_model.save_pretrained(ckpt_dir, save_function = self.accelerator.save)
        self.tokenizer.save_pretrained(ckpt_dir)

        return ckpt_dir
    
@dataclass
class SimpleTrainer(BaseTrainer):
    config: DictConfig
    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    train_dataset: Dataset

    def prepare(self):
        self.train_dataloader = self.accelerator.prepare(self.get_train_dataloader())
        self.optimizer = self.get_optimizer()

        # scheduler and math around the number of training steps
        # note the training dataloader needs to be prepared before grabing its length below
        # (cause its length will be shorter in multiprocess)
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        if self.config.max_train_steps == None:
            self.config.max_train_steps = (
                self.config.num_train_epochs * num_update_steps_per_epoch
            )
        else:
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / num_update_steps_per_epoch
            )

        if self.config.num_warmup_steps == None and self.config.warmup_ratio != None:
            self.config.num_warmup_steps = math.ceil(
                self.config.warmup_ratio * self.config.max_train_steps
            )

        self.lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )

        # prepare with `accelerator`
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

    def get_optimizer(self):
        # split weights in two groups, one with weight decay and the other not
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n,  p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        return optimizer

    def get_train_dataloader(self):
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None),
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            pin_memory=True,
            num_workers=4,
        )
        return train_dataloader

    @th.no_grad()
    def evaluate(self, ckpt_dir: str = None):
        pass

    def train(self):
        config = self.config
        accelerator = self.accelerator
        model = self.model
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        train_dataloader = self.train_dataloader

        total_batch_size = (
            config.per_device_train_batch_size
            * accelerator.num_processes
            * config.gradient_accumulation_steps
        )
        logging.info("***** Training *****")
        logging.info(f"  Num train examples = {len(self.train_dataset)}")
        logging.info(f"  Num epochs = {config.num_train_epochs}")
        logging.info(
            f"  Instantaneous batch size per device = {config.per_device_train_batch_size}"
        )
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(
            f"  Gradient accumulation steps = {config.gradient_accumulation_steps}"
        )
        logging.info(f"  Total optimization steps = {config.max_train_steps}")
        # only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(config.max_train_steps),
            disable=not accelerator.is_local_main_process,
        )
        completed_steps = 0
        for epoch in range(config.num_train_epochs):
            tr_loss = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    model.train()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                tr_loss += loss.item()

                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": tr_loss / (step + 1),
                        }
                    )
                    completed_steps += 1

                    if completed_steps >= config.max_train_steps:
                        logging.info(f"final step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            self.evaluate(ckpt_dir)
                        break
                    elif (
                        config.ckpt_step > 0 and completed_steps % config.ckpt_step == 0
                    ):
                        logging.info(f"step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            self.evaluate(ckpt_dir)

            if config.get("ckpt_epoch", True):
                logging.info(f"epoch {epoch}:")
                logging.info(f"  loss: {tr_loss/ (step + 1)}")
                ckpt_dir = self.save_ckpt(completed_epochs=epoch)
                if config.get("eval_epoch", False):
                    self.evaluate(ckpt_dir)

@dataclass
class TripletTrainer(SimpleTrainer):
    config: DictConfig
    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    train_dataset: Dict[str, Dataset]
    test_dataset: Dict[str, Dataset]

    @th.no_grad()
    def get_train_dataloader(self):
        data_collator = DataCollatorForTriplet(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None)
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.per_device_train_batch_size,
            shuffle = True,
            collate_fn = data_collator,
            pin_memory = True,
            num_workers = 4
        )
        return train_dataloader
    
    def train(self):
        config = self.config
        accelerator = self.accelerator
        model = self.model
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        train_dataloader = self.train_dataloader

        total_batch_size = (
            config.per_device_train_batch_size
            * accelerator.num_processes
            * config.gradient_accumulation_steps
        )
        logging.info("***** Training *****")
        logging.info(f"  Num train examples = {len(self.train_dataset)}")
        logging.info(f"  Num epochs = {config.num_train_epochs}")
        logging.info(
            f"  Instantaneous batch size per device = {config.per_device_train_batch_size}"
        )
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(
            f"  Gradient accumulation steps = {config.gradient_accumulation_steps}"
        )
        logging.info(f"  Total optimization steps = {config.max_train_steps}")
        # only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(config.max_train_steps),
            disable=not accelerator.is_local_main_process,
        )

        # fgm = FGM(model)
        completed_steps = 0
        for epoch in range(config.num_train_epochs):
            tr_loss = 0
            tr_loss_adv = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    model.train()

                    loss = model(train_batch=batch)
                    accelerator.backward(loss)

                    ######### fgm start ########
                    # fgm.attack()
                    # loss_adv = model(train_batch=batch)
                    # accelerator.backward(loss_adv)
                    # fgm.restore()

                    ######### fgm end ########

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                tr_loss += loss.item()
                # tr_loss_adv += loss_adv.item()

                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": tr_loss / (step + 1),
                            # "adv_loss": tr_loss_adv / (step + 1),
                        }
                    )
                    completed_steps += 1

                    if completed_steps >= config.max_train_steps:
                        logging.info(f"final step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        lr = lr_scheduler.get_last_lr()[0]
                        logging.info(f"  lr:{lr}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            pass
                        break
                    elif (
                        config.ckpt_step > 0 and completed_steps % config.ckpt_step == 0
                    ):
                        logging.info(f"step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            pass

            if config.get("ckpt_epoch", True):
                logging.info(f"epoch {epoch}:")
                logging.info(f"  loss: {tr_loss/ (step + 1)}")
                ckpt_dir = self.save_ckpt(completed_epochs=epoch)
                if config.get("eval_epoch", False):
                    pass

@dataclass
class Rerank2TripletTrainer(TripletTrainer):
    def get_train_dataloader(self):
        data_collator = DataCollatorForTriplet(
            self.tokenizer,
            pad_to_multiple_of = (8 if self.accelerator.use_fp16 else None),
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.per_device_train_batch_size,
            shuffle = True,
            collate_fn = data_collator,
            pin_memory = True,
            num_workers = 4
        )
        return train_dataloader

@dataclass
class RerankTrainerMultiFieldWithPointwise(SimpleTrainer):
    config: DictConfig
    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    train_dataset: Dict[str, Dataset]
    test_dataset: Dict[str, Dataset]

    @th.no_grad()
    def get_train_dataloader(self):
        data_collator = DataCollatorForRerankMultiFieldWithPointwiseLabel(
            tokenizer=self.tokenizer,
            neg_num=self.config.neg_num,
            pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None)
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.per_device_train_batch_size,
            shuffle = True,
            collate_fn = data_collator,
            pin_memory = True,
            num_workers = 16
        )
        return train_dataloader
    
    def train(self):
        config = self.config
        accelerator = self.accelerator
        model = self.model
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        train_dataloader = self.train_dataloader

        total_batch_size = (
            config.per_device_train_batch_size
            * accelerator.num_processes
            * config.gradient_accumulation_steps
        )
        logging.info("***** Training *****")
        logging.info(f"  Num train examples = {len(self.train_dataset)}")
        logging.info(f"  Num epochs = {config.num_train_epochs}")
        logging.info(
            f"  Instantaneous batch size per device = {config.per_device_train_batch_size}"
        )
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(
            f"  Gradient accumulation steps = {config.gradient_accumulation_steps}"
        )
        logging.info(f"  Total optimization steps = {config.max_train_steps}")
        # only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(config.max_train_steps),
            disable=not accelerator.is_local_main_process,
        )

        #fgm = FGM(model)
        completed_steps = 0
        for epoch in range(config.num_train_epochs):
            tr_loss = 0
            tr_loss_adv = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    model.train()

                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                tr_loss += loss.item()

                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": tr_loss / (step + 1),
                            "tmp_loss": loss.item()
                        }
                    )
                    completed_steps += 1

                    if completed_steps >= config.max_train_steps:
                        logging.info(f"final step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            pass
                        break
                    elif (
                        config.ckpt_step > 0 and completed_steps % config.ckpt_step == 0
                    ):
                        logging.info(f"step {completed_steps}:")
                        logging.info(f"  loss: {tr_loss/ (step + 1)}")
                        ckpt_dir = self.save_ckpt(completed_steps=completed_steps)
                        if config.get("eval_step", False):
                            pass

            if config.get("ckpt_epoch", True):
                logging.info(f"epoch {epoch}:")
                logging.info(f"  loss: {tr_loss/ (step + 1)}")
                ckpt_dir = self.save_ckpt(completed_epochs=epoch)
                if config.get("eval_epoch", False):
                    pass