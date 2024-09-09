import logging
import os
import json
import time
import yaml
import torch

from configs.opts import get_opts
from models.build.build_loader import build_dataset
from test import test_fn
from train import train_fn
from utils.sys_utils import init_distributed, set_random_seed
from models.build.build_model import build_model
from models.layers.clip import clip


logger = logging.getLogger(__name__)
if __name__ == '__main__':
    cfg = get_opts()
    init_distributed(0, cfg)
    yaml.dump(cfg, open(os.path.join(cfg.data.checkpoints_dir, 'config.yaml'), 'w'))
    logging.basicConfig(level=getattr(logging, cfg.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    set_random_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_weights = cfg.data.clip_weights
    pretrained_model, transform = clip.load(name=clip_weights)

    model = build_model(config=cfg,pretrained=pretrained_model)
    model = model.float()
    model = model.to(device)
    logger.info(f'Model total parameters: {sum(p.numel() for p in model.parameters()):,}')

    train_loader, valid_loader, test_loader = build_dataset(cfg,transform)
    # if cfg.decoder.decoder_type=="lstm":
    # model.load_state_dict(torch.load(
    #     '/media/hpc/13CE9BF66AC0DE98/CX/VC/Tracker4Cap/result/checkpoints/msrvtt/Track_True_AGE_True_checkpoint_2024-07-03T18:08:35/clip_l14_epochs_1_lr_0.0002_max_objects_1.ckpt'))
    # train_fn(cfg, model, train_loader, test_loader, device)
    model.load_state_dict(torch.load('/media/hpc/13CE9BF66AC0DE98/CX/VC/Tracker4Cap/result/checkpoint/msvd/Fusion_ablation/B=128/clip_l14_epochs_3_lr_0.0002_max_objects_3.ckpt'), strict=False)
    model.eval()
    test_fn(cfg, model, test_loader, device)

# --fusion_object--fusion_action