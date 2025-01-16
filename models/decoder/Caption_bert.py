import logging
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from easydict import EasyDict as edict

from models.layers.bert import BertSelfEncoder, BertLMPredictionHead

logger = logging.getLogger(__name__)
class CaptionHead(nn.Module):
    def __init__(
            self, hidden_dim,
            word_embedding_size, visual_feature_size,
            pretrained_embedding,
            max_v_len, max_t_len, hidden_size,
            vocab_size,fusion_object, fusion_action
    ):
        super(CaptionHead, self).__init__()
        self.fusion_object, self.fusion_action = fusion_object,fusion_action
        # sentence
        total_visual_dim = visual_feature_size
        if self.fusion_object:
            self.Uo = nn.Linear(hidden_dim, visual_feature_size)
            self.bo = nn.Parameter(torch.ones(visual_feature_size), requires_grad=True)
            self.wo = nn.Linear(visual_feature_size, 1)
            total_visual_dim+=visual_feature_size
        if self.fusion_action:
            self.Um = nn.Linear(hidden_dim, visual_feature_size)
            self.bm = nn.Parameter(torch.ones(visual_feature_size), requires_grad=True)
            self.wm = nn.Linear(visual_feature_size, 1)
            total_visual_dim += visual_feature_size
        setattr(self, 'linear_visual_layer', nn.Linear(total_visual_dim, visual_feature_size))
        self.cap_config = edict(
            word_vec_size=word_embedding_size,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=hidden_size,
            video_feature_size=visual_feature_size,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer layers
            num_attention_heads=8,
            share_wd_cls_weight=False,
            vocab_size=vocab_size,
            BOS_id=vocab_size - 2,
            EOS_id=vocab_size - 1,
            PAD_id=0
        )
        logger.debug("Caption Head Configuration: %s", self.cap_config)
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        self.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        self.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        assert torch.equal(self.cap_sa_decoder.word_embeddings.weight, self.prediction_head.decoder.weight)

    def Fusion(self, visual, vp_features, object_features,fusion_object=True,fusion_action=True,):
        if fusion_object:
            U_objs = self.Uo(object_features)
            attn_feat = visual.unsqueeze(2) + U_objs.unsqueeze(1) + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
            attn_weights = self.wo(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
            attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
            attn_objects = attn_weights * attn_feat
            attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
            features = torch.cat([visual, attn_objects], dim=-1)
        else:
           features = visual
        if fusion_action:
            U_motion = self.Um(vp_features)
            attn_feat = visual.unsqueeze(2) + U_motion.unsqueeze(1) + self.bm  # (bsz, sample_numb, sample_numb, hidden_dim)
            attn_weights = self.wm(torch.tanh(attn_feat))  # (bsz, sample_numb, sample_numb, 1)
            attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, sample_numb, 1)
            attn_motion = attn_weights * attn_feat
            attn_motion = attn_motion.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
            features = torch.cat([features, attn_motion], dim=-1)
        output = self.linear_visual_layer(features) if hasattr(self, 'linear_visual_layer') else features
        context = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        return context

    def forward(self, visual, objects, actions, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"
        if self.fusion_object or self.fusion_action:
            context_feats = self.Fusion(visual, objects, actions, self.fusion_object, self.fusion_action)
            context_feats=context_feats.unsqueeze(1)
            input_types = torch.concat(
                [
                    torch.full((visual.size(0), visual.size(1)),
                               fill_value=1, dtype=torch.long, device=visual.device),
                    torch.full((context_feats.size(0), context_feats.size(1)),
                               fill_value=0, dtype=torch.long, device=context_feats.device),
                    torch.full((input_ids.size(0), input_ids.size(1)),
                               fill_value=2, dtype=torch.long, device=input_ids.device)
                ], dim=1
            )
            visual_output = torch.cat([visual, context_feats], dim=1)
        else:
            input_types = torch.concat(
                [
                    torch.full((visual.size(0), visual.size(1)),
                               fill_value=1, dtype=torch.long, device=visual.device),
                    torch.full((input_ids.size(0), input_ids.size(1)),
                               fill_value=2, dtype=torch.long, device=input_ids.device)
                ], dim=1
            )
            visual_output = visual
        input_mask = torch.concat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        # logger.debug("GT  : %s", self.ids2text(input_ids))
        # logger.debug("Pred: %s", self.probability2text(prediction_scores))
        return prediction_scores


    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids: Union[np.ndarray, torch.Tensor]):
        if isinstance(gt_ids, np.ndarray) or isinstance(gt_ids, torch.Tensor):
            assert 0 < len(gt_ids.shape) <= 2, f"gt_ids should be a 1 dim or 2 dim array/tensor, got {gt_ids.shape}"
        else:
            raise ValueError("gt_ids should be np.ndarray or torch.Tensor")
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        if len(gt_ids.shape) == 1:
            return convert_ids_to_sentence(gt_ids.tolist())
        else:
            return [convert_ids_to_sentence(_gt_ids) for _gt_ids in gt_ids.tolist()]

def convert_ids_to_sentence(tokens):
    from models.layers.clip.clip import _tokenizer
    text = _tokenizer.decode(tokens)
    text_list = text.split(" ")
    new = []
    for i in range(len(text_list)):
        if i == 0:
            new.append(text_list[i].split(">")[-1])
        elif "<|endoftext|>" in text_list[i]:
            break
        else:
            new.append(text_list[i])
    return " ".join(new)