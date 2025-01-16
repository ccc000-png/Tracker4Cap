import logging
import os
import json
import time
import yaml
import torch
from configs.opts import get_opts
from models.build.build_loader import build_dataset
from utils.sys_utils import init_distributed, set_random_seed
from models.build.build_model import build_model
from models.layers.clip import clip

from configs.opts import TotalConfigs
from eval import eval_language_metrics

logger = logging.getLogger(__name__)

def test_fn(cfgs: TotalConfigs, model, loader, device):
    print('##############n_vocab is {}##############'.format(model.caption_head.cap_config.vocab_size))
    checkpoint = {
        "epoch": -1,
        "model_config": model.module if
        hasattr(model, 'module') else model
    }
    metrics = eval_language_metrics(checkpoint, loader, cfgs, model=model, device=device, eval_mode='test')

    logger.info('\t>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
                format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                       metrics['CIDEr'] * 100))
    log_stats = {**{f'[test{k}': v for k, v in metrics.items()},
                 }
    with open(cfgs.train.evaluate_dir, "a") as f:
        f.write(json.dumps(log_stats) + '\n')
    print('===================Testing is finished====================')

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

    model.load_state_dict(torch.load(cfg.train.save_checkpoints), strict=False)
    model.eval()
    test_fn(cfg, model, test_loader, device)