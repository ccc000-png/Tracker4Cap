import json
import logging

from eval import eval_language_metrics

logger = logging.getLogger(__name__)
def test_fn(cfgs, model, loader, device):
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

