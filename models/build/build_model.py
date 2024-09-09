import json
import logging
import math
import pickle
import random

import torch
from torch import nn

from models.decoder.Caption_bert import CaptionHead

from models.encoders.AGE import AGE, AGE_base
from models.encoders.AugEncoderLayer import AugEncoder
from models.encoders.FMOT import FMOT
from models.encoders.transformer import Transformer

logger = logging.getLogger(__name__)
def build_model(config, pretrained):
    model = Track4Cap(config, pretrained)
        # if config.train.save_checkpoints_path:
        #     model.load_state_dict(torch.load(config.train.save_checkpoints_path))
    return model

class Track4Cap(nn.Module):
    def __init__(self, cfgs,pretrained_model):
        super().__init__()
        self.Track = cfgs.Track
        self.Age = cfgs.Age
        self.cfgs = cfgs
        self.query_embed = nn.Embedding(cfgs.encoder.track_objects, cfgs.encoder.hidden_dim)
        if cfgs.Track:
            ObjectEncoder = FMOT(d_model=cfgs.encoder.hidden_dim)
        elif cfgs.Base==1:
            ObjectEncoder = Transformer(d_model=cfgs.encoder.hidden_dim)
        elif cfgs.Base==2:
            ObjectEncoder = BiLstm(d_model=cfgs.encoder.hidden_dim)
        if cfgs.Age:
            ActionEncoder = AGE(d_model=cfgs.encoder.hidden_dim)
        else:
            ActionEncoder = AGE_base(d_model=cfgs.encoder.hidden_dim)

        self.AugInformation = AugEncoder(ObjectEncoder= ObjectEncoder,
                                         ActionEncoder=ActionEncoder,
                                           max_objects= cfgs.encoder.track_objects,
                                           visual_dim = cfgs.encoder.visual_dim,
                                           object_dim = cfgs.encoder.object_dim,
                                           hidden_dim = cfgs.encoder.hidden_dim)
        '''2.字幕头'''
        self.visual_encoder = pretrained_model.visual
        state_dict = pretrained_model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        self.vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}

        self.caption_head = CaptionHead(hidden_dim = cfgs.encoder.hidden_dim,
                                        word_embedding_size=transformer_width,
                                        visual_feature_size=embed_dim,
                                        pretrained_embedding=pretrained_embedding,
                                        max_v_len=cfgs.sample_numb + 1,
                                        max_t_len=cfgs.decoder.max_caption_len,  # 77
                                        hidden_size=embed_dim,
                                        vocab_size=self.vocab_size,
                                        fusion_object = cfgs.fusion_object,
                                        fusion_action = cfgs.fusion_action,)

    def forward(self, global_visual, input_ids, input_mask):
        query_pos = self.query_embed.weight
        aug_object_features, aug_action_features= self.AugInformation(
            visual = global_visual,
            objects = None,
            query_pos = query_pos
        )

        prediction_scores = self.caption_head(
            global_visual,
            aug_object_features,
            aug_action_features,
            input_ids,
            input_mask,
        )
        return {"prediction_scores":prediction_scores}
