# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 16:02
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import argparse
import logging
import os

import torch
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from ipaddress import ip_address, IPv4Address
import yaml
from yaml.loader import SafeLoader

logger = logging.getLogger(__name__)

CUSTOM_CONFIG_REGISTRY = Registry("CUSTOM_CONFIG")
CUSTOM_CONFIG_CHECK_REGISTRY = Registry("CUSTOM_CONFIG_CHECK")

_C = CfgNode()

# project info
_C.INFO = CfgNode()
_C.INFO.PROJECT_NAME = None
_C.INFO.EXPERIMENT_NAME = "experiment"

# system config
_C.distributed = True
_C.init_method = "tcp://localhost:2222"
_C.num_gpu = torch.cuda.device_count()
_C.gpu_devices = list(range(torch.cuda.device_count()))
_C.num_shards = 1
_C.shard_id = 0
_C.deterministic = True
_C.seed = 222

# log config
_C.LOG = CfgNode()
_C.LOG.DIR = None
_C.LOG.LOGGER_FILE = "logger.log"
_C.loglevel = "info"
_C.LOG.LOGGER_CONSOLE_COLORFUL = True

# build config for base dataloader
_C.data = CfgNode()
_C.data.DATASET = CfgNode()
_C.data.DATASET.NAME = "MSRVTTCaptioningDatasetForCLIP"
_C.data.LOADER = CfgNode()
_C.data.LOADER.COLLATE_FN = None
_C.data.LOADER.BATCH_SIZE = 1
_C.data.LOADER.NUM_WORKERS = 1
_C.data.LOADER.SHUFFLE = True
_C.data.LOADER.PREFETCH_FACTOR = 2
_C.data.LOADER.MULTIPROCESSING_CONTEXT = "fork"

# msrvtt
_C.data.DATASET.MSRVTT = CfgNode()
_C.data.DATASET.MSRVTT.VIDEO_ROOT = "/media/hpc/39C3AC34579106FA/CX/CoCap/dataset/msrvtt/videos_h264_keyint_60"
_C.data.DATASET.MSRVTT.METADATA = "/media/hpc/39C3AC34579106FA/CX/CoCap/dataset/msrvtt/MSRVTT_data.json"
_C.data.DATASET.MSRVTT.VIDEO_READER = "read_frames_compressed_domain"
_C.data.DATASET.MSRVTT.MAX_FRAMES = 8
_C.data.DATASET.MSRVTT.VIDEO_SIZE = (224, 224)
_C.data.DATASET.MSRVTT.MAX_WORDS = 15
_C.data.DATASET.MSRVTT.UNFOLD_SENTENCES = True
# msvd
_C.data.DATASET.MSVD = CfgNode()
_C.data.DATASET.MSVD.VIDEO_ROOT = None
_C.data.DATASET.MSVD.METADATA = None
_C.data.DATASET.MSVD.VIDEO_READER = None
_C.data.DATASET.MSVD.MAX_FRAMES = None
_C.data.DATASET.MSVD.VIDEO_SIZE = None
_C.data.DATASET.MSVD.MAX_WORDS = None
_C.data.DATASET.MSVD.UNFOLD_SENTENCES = None
# vatex
_C.data.DATASET.VATEX = CfgNode()
_C.data.DATASET.VATEX.VIDEO_ROOT = None
_C.data.DATASET.VATEX.METADATA = None
_C.data.DATASET.VATEX.VIDEO_READER = None
_C.data.DATASET.VATEX.MAX_FRAMES = None
_C.data.DATASET.VATEX.VIDEO_SIZE = None
_C.data.DATASET.VATEX.MAX_WORDS = None
_C.data.DATASET.VATEX.UNFOLD_SENTENCES = None

# h265 config for h265 video readers
_C.CV_CONFIG = CfgNode()
_C.CV_CONFIG.NUM_GOP = 1
_C.CV_CONFIG.NUM_MV = 59
_C.CV_CONFIG.NUM_RES = 59
_C.CV_CONFIG.WITH_RESIDUAL = True
_C.CV_CONFIG.USE_PRE_EXTRACT = False
_C.CV_CONFIG.SAMPLE = "rand"

# build config for base model
_C.MODEL = CfgNode()
_C.MODEL.NAME = None
_C.MODEL.PARALLELISM = "ddp"
_C.MODEL.DDP = CfgNode()
_C.MODEL.DDP.FIND_UNUSED_PARAMETERS = False

_C.MODEL.COCAP = CfgNode()
_C.MODEL.COCAP.PRETRAINED_CLIP = "/media/hpc/13CE9BF66AC0DE98/CX/VC/Tracker4Cap/model_zoo/clip_model/ViT-L-14.pt"
_C.MODEL.COCAP.MOTION_DROPOUT_PROB = 0.2
_C.MODEL.COCAP.RESIDUAL_DROPOUT_PROB = 0.2
_C.MODEL.COCAP.MOTION_ENCODER = CfgNode()
_C.MODEL.COCAP.MOTION_ENCODER.N_LAYERS = 2
_C.MODEL.COCAP.MOTION_ENCODER.PATCH_SIZE = 8
_C.MODEL.COCAP.MOTION_ENCODER.N_HEADS = 8
_C.MODEL.COCAP.RESIDUAL_ENCODER = CfgNode()
_C.MODEL.COCAP.RESIDUAL_ENCODER.N_LAYERS = 2
_C.MODEL.COCAP.RESIDUAL_ENCODER.PATCH_SIZE = 64
_C.MODEL.COCAP.RESIDUAL_ENCODER.N_HEADS = 8
_C.MODEL.COCAP.ACTION_ENCODER = CfgNode()
_C.MODEL.COCAP.ACTION_ENCODER.N_LAYERS = 1
_C.MODEL.COCAP.ACTION_ENCODER.N_HEADS = 8
_C.MODEL.COCAP.TASK_TYPE = "captioning"


# optimizer
_C.OPTIMIZER = CfgNode()
_C.OPTIMIZER.NAME = "BertAdam"
_C.OPTIMIZER.PARAMETER = CfgNode(new_allowed=True)
_C.OPTIMIZER.PARAMETER.lr = 1e-4
_C.OPTIMIZER.PARAMETER.warmup = 0.1
_C.OPTIMIZER.PARAMETER.schedule = "warmup_constant"
_C.OPTIMIZER.PARAMETER.weight_decay = 0.01
_C.OPTIMIZER.PARAMETER.max_grad_norm = 1.0

# scheduler
_C.SCHEDULER = CfgNode()
_C.SCHEDULER.NAME = None

# build config for loss
_C.LOSS = CfgNode()
_C.LOSS.NAME = "LabelSmoothingLoss"
_C.LOSS.MultiObjectiveLoss = CfgNode()
_C.LOSS.MultiObjectiveLoss.LOSSES = []
_C.LOSS.MultiObjectiveLoss.WEIGHT = None

# build config for meter
_C.METER = CfgNode()
_C.METER.NAME = None

# trainer
_C.TRAINER = CfgNode()
_C.bert = CfgNode()
_C.train = CfgNode()
_C.train.evaluate_dir = '/media/hpc/13CE9BF66AC0DE98/CX/VC/Tracker4Cap/Cocap/results'
_C.TRAINER.NAME = "TrainerBase"
# trainer base
_C.TRAINER.TRAINER_BASE = CfgNode()
_C.TRAINER.TRAINER_BASE.TEST_ENABLE = True
_C.TRAINER.TRAINER_BASE.TRAIN_ENABLE = True
_C.TRAINER.TRAINER_BASE.EPOCH = 20
_C.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS = 1
_C.TRAINER.TRAINER_BASE.RESUME = None
_C.TRAINER.TRAINER_BASE.AUTO_RESUME = False
_C.TRAINER.TRAINER_BASE.CLIP_NORM = None
_C.TRAINER.TRAINER_BASE.SAVE_FREQ = 1
_C.TRAINER.TRAINER_BASE.LOG_FREQ = 1
_C.TRAINER.TRAINER_BASE.AMP = False
_C.TRAINER.TRAINER_BASE.DEBUG = False
_C.TRAINER.TRAINER_BASE.WRITE_HISTOGRAM = False
_C.TRAINER.TRAINER_BASE.WRITE_PROFILER = False

_C.TRAINER.CAPTION_TRAINER = CfgNode()
_C.TRAINER.CAPTION_TRAINER.TASK_TYPE = None
_C.TRAINER.CAPTION_TRAINER.CLIP_LR = 1e-6
_C.TRAINER.CAPTION_TRAINER.LR_DECAY_GAMMA = 0.95

base_config = _C.clone()
base_config.freeze()


def check_config(cfg: CfgNode):
    # default check config
    if cfg.LOG.DIR is None:
        info = [i for i in [cfg.INFO.PROJECT_NAME, cfg.INFO.EXPERIMENT_NAME] if i]
        if info:  # not empty
            cfg.LOG.DIR = os.path.join("log", "_".join(info))
        else:
            cfg.LOG.DIR = os.path.join("log", "default")
    assert cfg.MODEL.PARALLELISM.lower() in {"dp", "ddp", "fsdp"}, "MODEL.PARALLELISM should be one of {dp, ddp, fsdp}"
    assert not cfg.SYS.MULTIPROCESS or cfg.SYS.NUM_GPU > 0, "At least 1 GPU is required to enable ddp."
    assert cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS > 0, "gradient accumulation step should greater than 0."
    assert cfg.LOSS.MultiObjectiveLoss.WEIGHT is None or \
           len(cfg.LOSS.MultiObjectiveLoss.WEIGHT) == len(cfg.LOSS.MultiObjectiveLoss.LOSSES)


def get_config():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Which project to run. The config will be loaded according to project. "
                                          "Load all registered configs if project is not set.",
                        default=None)
    parser.add_argument("--cfg", "-c",
                        help="path to the additional config file",
                        default=None,
                        type=str)
    parser.add_argument("--debug",
                        help="set trainer to debug mode",
                        action="store_true")
    parser.add_argument("opts",
                        help="see config/custom_config.py for all options",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # build base config and custom config
    cfg = base_config.clone()
    cfg.defrost()
    return cfg


if __name__ == '__main__':
    print(get_config())
