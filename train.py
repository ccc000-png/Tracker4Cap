import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import torch.distributed as dist
from tqdm import tqdm

from eval import eval_language_metrics
from models.build.build_loss import build_loss
from models.build.build_optimizer import prep_optimizer
from configs.opts import TotalConfigs


logger = logging.getLogger(__name__)
def train_fn(cfgs: TotalConfigs, model: nn.Module, train_loader, valid_loader, device):
    gradient_accumulation_step = cfgs.bert.gradient_accumulation_steps
    num_train_optimization_steps = (int(len(train_loader) + gradient_accumulation_step - 1)
                                    / gradient_accumulation_step) * cfgs.train.max_epochs
    cfgs.bert.t_total = num_train_optimization_steps
    optimizer, scheduler = prep_optimizer(cfgs, model)
    loss_func = build_loss(cfgs)
    best_score, cnt = None, 0
    global_step =  0 * (len(train_loader) // gradient_accumulation_step)

    for epoch in range(cfgs.train.max_epochs):
        logger.debug(f"Epoch {epoch + 1}/{cfgs.train.max_epochs}")
        model.train()
        torch.cuda.empty_cache()
        bar = train_loader = tqdm(train_loader,
                                  desc=f"Train: {epoch + 1}/{cfgs.train.max_epochs}",
                                  dynamic_ncols=True,
                                  disable=dist.is_initialized() and dist.get_rank() != 0)
        loss_total = 0.
        logger.debug("Running train epoch for-loop...")
        for cur_step, (
                feature2ds, input_ids, input_labels, input_mask, captions,
                video_id) in enumerate(train_loader):
            feature2ds = feature2ds.to(device)
            # objects = objects.to(self.device)

            input_ids = input_ids.to(device)
            input_labels = input_labels.to(device)
            input_mask = input_mask.to(device)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(feature2ds, input_ids, input_mask)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference Time: {inference_time:.6f} seconds")
            loss_meta = loss_func(input_labels, outputs)
            # backward
            if isinstance(loss_meta, dict):
                loss = sum([v for _, v in loss_meta.items()])
            else:
                loss = loss_meta
            loss /= gradient_accumulation_step
            loss.backward()
            loss_total += loss.detach()
            if gradient_accumulation_step > 1:
                bar.set_postfix(
                    {"Accumulation Step": (cur_step + 1) % gradient_accumulation_step}
                )

            if (cur_step + 1) % gradient_accumulation_step == 0:
                # optimize
                optimizer.step()
                optimizer.zero_grad()
            # summary
            with torch.no_grad():
                if cur_step % cfgs.data.log_freq == 0:
                    if dist.is_initialized():
                        dist.all_reduce(loss)
                        loss = loss / dist.get_world_size()
                        logger.debug(
                            f"loss (rank {dist.get_rank()}, step {global_step}): {loss.cpu().detach().numpy()}"
                        )
                    else:
                        logger.debug(f"loss (step {global_step}): {loss.cpu().detach().numpy()}")
            loss_total = 0.
            global_step += 1

        logger.debug("Train epoch for-loop finished.")
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.empty_cache()

        logger.debug("Running valid")
        model.eval()
        checkpoint = {
            "epoch": epoch,
            "model_config": model.module if
            hasattr(model, 'module') else model
        }
        metrics = eval_language_metrics(checkpoint, valid_loader, cfgs, model=model, device=device,
                                        eval_mode='valid')

        if not dist.is_initialized() or dist.get_rank() == 0:
            ckpt_path = cfgs.train.save_checkpoints_path
            # torch.save(self.model.state_dict(), ckpt_path)
            cider_score = metrics['CIDEr']
            if best_score is None or cider_score > best_score:
                best_score = cider_score
                torch.save(model.state_dict(), ckpt_path)

            logger.info('\t>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
                        format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                               metrics['CIDEr'] * 100))

            logger.info('\t>>>  Best_CIDEr: {:.2f}'.format(best_score * 100))

            log_stats = {**{f'[EPOCH{epoch + 1}]_test{k}': v for k, v in metrics.items()},
                         }
            with open(cfgs.train.evaluate_dir, "a") as f:
                f.write(json.dumps(log_stats) + '\n')

    print('===================Training is finished====================')
    return model