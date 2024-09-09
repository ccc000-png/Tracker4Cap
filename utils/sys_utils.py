import datetime
import hashlib
import os
import pickle
import typing
import torch
import time
import random
import itertools
import numpy as np
import logging
from typing import *

import torch.distributed as dist

logger = logging.getLogger(__name__)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    logger.debug("Manual seed is set")

def init_distributed(proc: int, cfg):
    if cfg.distributed:  # initialize multiprocess
        word_size = cfg.num_gpu * cfg.num_shards
        rank = cfg.num_gpu * cfg.shard_id + proc
        dist.init_process_group(backend="nccl", init_method=cfg.init_method, world_size=word_size, rank=rank)
        gpu_devices = list(range(torch.cuda.device_count()))
        torch.cuda.set_device(gpu_devices[proc])