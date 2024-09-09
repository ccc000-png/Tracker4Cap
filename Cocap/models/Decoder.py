import numpy as np
import torch
from torch import nn

from eval import convert_ids_to_sentence
from models.layers.bert import BertSelfEncoder, BertLMPredictionHead
from easydict import EasyDict as edict

class CaptionHead(nn.Module):

    def __init__(
            self,
            word_embedding_size: int, visual_feature_size: int,
            max_v_len: int, max_t_len: int, hidden_size: int,
            vocab_size: int,
    ):
        super(CaptionHead, self).__init__()
        self.model_network = "Self"
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
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        # debug output cfgs

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids):

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

    def forward(self, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"

        input_types = torch.concat(
            [
                torch.full((visual_output["feature_context"].size(0), visual_output["feature_context"].size(1)),
                           fill_value=1, dtype=torch.long, device=visual_output["feature_context"].device),
                torch.full((visual_output["feature_action"].size(0), visual_output["feature_action"].size(1)),
                           fill_value=0, dtype=torch.long, device=visual_output["feature_action"].device),
                torch.full((input_ids.size(0), input_ids.size(1)),
                           fill_value=2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_output = torch.cat([visual_output["feature_context"], visual_output["feature_action"]], dim=1)
        input_mask = torch.concat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        return prediction_scores

    @classmethod
    def from_pretrained(
            cls, pretrained_model, max_v_len, max_t_len
    ):

        state_dict = pretrained_model.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]

        head = cls(
            word_embedding_size=transformer_width,
            visual_feature_size=embed_dim,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=embed_dim,
            vocab_size=vocab_size,
        )
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        head.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        head.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        head.cap_sa_decoder.word_embeddings.to('cuda:0')
        head.prediction_head.decoder.to('cuda:0')
        assert torch.equal(head.cap_sa_decoder.word_embeddings.weight, head.prediction_head.decoder.weight)
        return head