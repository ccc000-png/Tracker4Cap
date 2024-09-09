from typing import Dict

import torch
from torch import nn

from Cocap.models.Decoder import CaptionHead
from Cocap.models.Encoder import CompressedVideoTransformer


class CoCap(nn.Module):
    def __init__(self, cfg, pretrained_model):
        super().__init__()
        cfg_model = cfg.MODEL.COCAP

        self.compressed_video_transformer = CompressedVideoTransformer.from_pretrained(
            pretrained_clip=pretrained_model,
            # motion
            motion_patch_size=cfg_model.MOTION_ENCODER.PATCH_SIZE,
            motion_layers=cfg_model.MOTION_ENCODER.N_LAYERS,
            motion_heads=cfg_model.MOTION_ENCODER.N_HEADS,
            # residual
            residual_patch_size=cfg_model.RESIDUAL_ENCODER.PATCH_SIZE,
            residual_layers=cfg_model.RESIDUAL_ENCODER.N_LAYERS,
            residual_heads=cfg_model.RESIDUAL_ENCODER.N_HEADS,
            # action
            action_layers=cfg_model.ACTION_ENCODER.N_LAYERS,
            action_heads=cfg_model.ACTION_ENCODER.N_HEADS,
            n_bp=cfg.CV_CONFIG.NUM_MV
        )

        self.dropout_motion = nn.Dropout(cfg_model.MOTION_DROPOUT_PROB)
        self.dropout_residual = nn.Dropout(cfg_model.RESIDUAL_DROPOUT_PROB)

        self.task_type = cfg_model.TASK_TYPE

        if self.task_type == "captioning":
            self.caption_head = CaptionHead.from_pretrained(
                pretrained_model=pretrained_model,
                max_t_len=cfg.data.DATASET.MSRVTT.MAX_WORDS, max_v_len=cfg.CV_CONFIG.NUM_GOP * 2
            )
        else:
            raise ValueError("Task type not supported: %s" % self.task_type)

    def forward(self, inputs):
        """

        :param inputs:
            video:
                iframe:         batch_size n_gop c h w
                motion_vector:  batch_size n_gop n_mv c=4|9 h/4 w/4
                residual:       batch_size n_gop n_res c h w
                input_mask_gop: batch_size n_gop
                input_mask_mv:  batch_size n_gop n_mv
        :return:
        """
        device = inputs["input_ids"].device
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = inputs["video"]["motion_vector"]
            residual = inputs["video"]["residual"] / 128 - 1  # for saving memory
            bp_type_ids = inputs["video"]["type_ids_mv"]

            motion = self.dropout_motion(motion)
            residual = self.dropout_residual(residual)
            compressed_visual_features = self.compressed_video_transformer(
                iframe=iframe.to(device),
                motion=motion.to(device),
                residual=residual.to(device),
                bp_type_ids=bp_type_ids.to(device)
            )
        else:
            # reuse pre-extracted visual features
            compressed_visual_features = inputs["visual_output"]

        if self.task_type == "captioning":
            prediction_scores = self.caption_head(
                compressed_visual_features,
                inputs["input_ids"],
                inputs["input_mask"],
            )
            return {"prediction_scores": prediction_scores, "visual_output": compressed_visual_features}
        else:
            raise ValueError("Task type not supported: %s" % self.task_type)