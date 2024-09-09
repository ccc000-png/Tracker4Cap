import logging
import os
import json
import time
import yaml
import torch

from Cocap.config.base import get_config
from Cocap.models.Cocap import CoCap
from Cocap.test import test_fn
# from Cocap.configs.opts import get_opts
from Cocap.dataloader.build_loader import build_dataset

from Cocap.train import train_fn
from configs.opts import get_timestamp
from utils.sys_utils import init_distributed, set_random_seed

from models.layers.clip import clip


logger = logging.getLogger(__name__)
if __name__ == '__main__':
    cfg = get_config()
    init_distributed(0, cfg)
    logging.basicConfig(level=getattr(logging, cfg.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    set_random_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_weights = cfg.MODEL.COCAP.PRETRAINED_CLIP
    pretrained_model, transform = clip.load(name=clip_weights)

    model = CoCap(cfg,pretrained_model)
    model = model.float()
    model = model.to(device)
    logger.info(f'Model total parameters: {sum(p.numel() for p in model.parameters()):,}')

    train_loader, test_loader = build_dataset(cfg,transform)
    cfg.train.checkpoints_dir = os.path.join(cfg.train.evaluate_dir,
                                        "MSRVTT/Cocap_{}".format(get_timestamp()))

    if not os.path.exists(cfg.train.checkpoints_dir):
        os.makedirs(cfg.train.checkpoints_dir)
    cfg.train.evaluate_dir = os.path.join(cfg.train.checkpoints_dir, 'evaluate.txt')

    # train_fn(cfg, model, train_loader, test_loader, device)
    # model.load_state_dict(torch.load('/media/hpc/13CE9BF66AC0DE98/CX/VC/Tracker4Cap/Cocap/config/msrvtt_captioning_pytorch_model.bin'), strict=False)
    model.eval()
    test_fn(cfg, model, test_loader, device)

# --fusion_object--fusion_action