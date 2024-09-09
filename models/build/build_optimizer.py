import logging
import math
import torch
from torch import optim, nn

from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim import lr_scheduler

from models.layers.bert import BertLayerNorm

logger = logging.getLogger(__name__)

def prep_optimizer(cfg, model):
    # based on:
    # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
    if hasattr(model, 'module'):
        model = model.module

    decay = set()
    no_decay = set()
    # pretrained_modules = [
    #     "caption_head.cap_sa_decoder.word_embeddings",
    #     "caption_head.prediction_head.decoder",
    # ]
    pretrained_modules = [
        "compressed_video_transformer.rgb_encoder.conv1",
        "compressed_video_transformer.rgb_encoder.class_embedding",
        "compressed_video_transformer.rgb_encoder.positional_embedding",
        "compressed_video_transformer.rgb_encoder.ln_pre",
        "compressed_video_transformer.rgb_encoder.transformer",
        "compressed_video_transformer.rgb_encoder.ln_post",
        "compressed_video_transformer.rgb_encoder.proj",
        "caption_head.cap_sa_decoder.word_embeddings",
        "caption_head.prediction_head.decoder",
    ]
    encoder = ['Encoder_layer.entity_level',]
    whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
    blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding, BertLayerNorm)
    # param_dict = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # param_dict[pn] = p
            if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
                no_decay.add(fpn)
            elif pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("bias_hh_l0") or pn.endswith("bias_hh_l0_reverse"):
                no_decay.add(fpn)
            elif pn.endswith("bias_ih_l0") or pn.endswith("bias_ih_l0_reverse"):
                no_decay.add(fpn)
            elif pn.endswith("proj"):
                decay.add(fpn)
            elif pn.endswith("projection"):
                decay.add(fpn)
            elif fpn.endswith("embedding"):
                no_decay.add(fpn)
            elif pn.endswith("weight_hh_l0") or pn.endswith("weight_hh_l0_reverse"):
                decay.add(fpn)
            elif pn.endswith("weight_ih_l0") or pn.endswith("weight_ih_l0_reverse"):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif pn.endswith("b") or pn.endswith("bo") or pn.endswith("bm") or pn.endswith("bv") or pn.endswith("bos"):
                decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    all = [pn for pn in sorted(list(param_dict.keys())) if
           any(pn.startswith(p_pn) for p_pn in encoder)]
    all_nodecay = [pn for pn in sorted(list(no_decay)) if
                   any(pn.startswith(p_pn) for p_pn in encoder)]
    all_decay = [pn for pn in sorted(list(decay)) if
                   any(pn.startswith(p_pn) for p_pn in encoder)]
    all2=all_decay+all_nodecay
    last = [pn for pn in sorted(all) if pn not in all2]
    for pn in last:
        decay.add(pn)
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),)

    pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                           any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
    not_pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                               not any(pn.startswith(p_pn) for p_pn in pretrained_modules)]

    decay_param = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_pretrained_param = [param_dict[pn] for pn in sorted(list(pretrained_no_decay))]
    no_decay_not_pretrained_param = [param_dict[pn] for pn in sorted(list(not_pretrained_no_decay))]

    optimizer_grouped_parameters = [
        {"params": decay_param},
        {"params": no_decay_pretrained_param, "weight_decay": 0.0, "lr": cfg.bert.clip_lr},
        {"params": no_decay_not_pretrained_param, "weight_decay": 0.0}
    ]

    warmup_epoch = int(cfg.bert.warmup * cfg.train.max_epochs)
    optimizer = build_optimizer(cfg, optimizer_grouped_parameters)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1 if epoch < warmup_epoch
        else cfg.bert.lr_decay_gamma ** (epoch - warmup_epoch)
    )

    return optimizer, scheduler

def build_optimizer(cfg, params) -> optim.Optimizer:
    logger.debug("Parameter for optimizer BertAdam is set")
    return BertAdam(params, lr=cfg.bert.lr, warmup=cfg.bert.warmup, t_total=cfg.bert.t_total, schedule=cfg.bert.schedule,
                weight_decay=cfg.bert.weight_decay,
                 max_grad_norm=cfg.bert.max_grad_norm)

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x / warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)

SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=1e-4, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # next_m.mul_(beta1).add_(1 - beta1, grad) --> pytorch 1.7
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad) --> pytorch 1.7
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss
