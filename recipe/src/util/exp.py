import logging
import math
import os
import random
import time
from typing import List, Union

import datasets
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf

def create_exp_folder(exp_dir, folder_name: str = "", comment: str = None):
    timestr = time.strftime("%m%d_%I%M%S")
    if comment:
        time_str = comment + "-" + timestr
    exp_dir = os.path.join(exp_dir, folder_name, timestr)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, "checkpoint"))

    return exp_dir

def set_seed(seed: int = None):
    if seed == None:
        seed = int(math.modf(time.time())[0] * 1000000)
        logging.info(f"Setting new random seed {seed}")
    else:
        logging.info(f"Setting random seed {seed}")
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except:
        ...

    try:
        import torch as th
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    except:
        ...
    return seed

def load_config(cfg_file: Union[str, List[str]] = None):
    cli_cfg = OmegaConf.from_cli()
    if(cli_cfg.get("config") != None):
        exp_cfg = OmegaConf.load(cli_cfg.config)
    elif(cfg_file != None):
        if isinstance(cfg_file, str):
            exp_cfg = OmegaConf.load(cfg_file)
        elif isinstance(cfg_file, List):
            exp_cfg = OmegaConf.load(cfg_file[0])
            for file in cfg_file[1:]:
                exp_cfg = OmegaConf.merge(exp_cfg, OmegaConf.load(file))
    else:
        exp_cfg = OmegaConf.load("./config/conf.yaml")
    exp_cfg = OmegaConf.merge(exp_cfg, cli_cfg)

    return exp_cfg

def setup(cfg_file: str = None, exp_folder: bool = True, mixed_precision: str = None):
    logging.basicConfig(format="%(message)s", level = logging.DEBUG, force = True)

    exp_cfg = load_config(cfg_file = cfg_file)

    if exp_cfg.get("debug", False) == True:
        logging.basicConfig(format="%(message)s", level = logging.DEBUG, force = True)
        logging.debug("***** Debugging *****")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        accelerator = Accelerator(
            split_batches = exp_cfg.get("split_bathces", False),
            gradient_accumulation_steps = exp_cfg.get("gradient_accumulation_steps", 1),
            mixed_precision = "no",
            kwargs_handlers = [ddp_kwargs],
        )
    else:
        accelerator = Accelerator(
            split_batches = exp_cfg.get("split_bathces", False),
            gradient_accumulation_steps = exp_cfg.get("gradient_accumulation_steps", 1),
            mixed_precision=mixed_precision
        )
    
    logging.info(accelerator.state)

    if(accelerator.is_local_main_process):
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        exp_cfg.random_seed = set_seed(exp_cfg.get("random_seed"))
        
        if(exp_folder and not exp_cfg.get("debug")):
            if(not exp_cfg.get("comment")):
                exp_cfg.comment = exp_cfg.get("model_name", "").split("/")[-1]
            exp_cfg.exp_dir = create_exp_folder(
                exp_cfg.exp_dir,
                folder_name = exp_cfg.folder_name,
                comment = exp_cfg.comment,
            )
            OmegaConf.save(config = exp_cfg, f = os.path.join(exp_cfg.exp_dir, "conf.yaml"))
            
            log_file = os.path.join(exp_cfg.exp_dir, "main.log")
            logging.basicConfig(
                format = "%(message)s",
                level = logging.INFO,
                handlers=[
                    logging.FileHandler(log_file, mode = "x", encoding = 'utf-8'),
                    logging.StreamHandler(),
                ],
                force = True
            )
            logging.info(f"Experiment folder: {exp_cfg.exp_dir}")
        else:
            exp_cfg.exp_dir = None
    else:
        logging.basicConfig(format = "%(message)s", level = logging.ERROR, force = True)
        datasets.utils.disable_progress_bar()
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        set_seed(exp_cfg.get("random_seed"))
        exp_cfg.exp_dir = None
    return exp_cfg, accelerator