import json
import os
import random
import time
from collections import defaultdict

import PIL
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

from Cocap.dataloader.transforms import DictCenterCrop, DictRandomHorizontalFlip, DictNormalize
from Cocap.dataloader.video_readers import VIDEO_READER_REGISTRY
from Cocap.dataloader.video_text_base import get_video
from models.dataset.utils import read_frames_decord
from models.layers.clip import clip


class CaptionDatasetE2E(Dataset):
    def __init__(self, cfgs, mode,transform):
        """1.获取文本信息"""
        self.mode = mode

        self.video_root = cfgs.data.DATASET.MSRVTT.VIDEO_ROOT
        self.max_words = cfgs.data.DATASET.MSRVTT.MAX_WORDS
        self.max_frames = cfgs.data.DATASET.MSRVTT.MAX_FRAMES

        self.unfold_sentences = cfgs.data.DATASET.MSRVTT.UNFOLD_SENTENCES  # only affect the train split
        self.height, self.width = cfgs.data.DATASET.MSRVTT.VIDEO_SIZE
        self.sentences = []  # (vid, [sentence, ...])
        self.h265_cfg = cfgs.CV_CONFIG
        metadata = json.load(open(cfgs.data.DATASET.MSRVTT.METADATA, 'r'))
        video_ids = [metadata['videos'][idx]['video_id'] for idx in range(len(metadata['videos']))]
        all_split_video_ids = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497],
                               "test": video_ids[6513 + 497:]}

        split_video_ids = all_split_video_ids[mode].copy()

        vid2sentence = defaultdict(list)
        for item in tqdm(metadata["sentences"]):
            if item["video_id"] in split_video_ids:
                vid2sentence[item["video_id"]].append(item["caption"])
        self.sentences = list(vid2sentence.items())

        # self.sentences = self.sentences[:50000]
        self.video_reader = VIDEO_READER_REGISTRY.get(cfgs.data.DATASET.MSRVTT.VIDEO_READER)
        # transforms
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if mode == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif mode == "test":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError

        if mode == "test":
            json_ref = {k: [] for k in all_split_video_ids[mode]}
            for sentence in tqdm(metadata["sentences"]):
                if sentence["video_id"] in json_ref:
                    json_ref[sentence["video_id"]].append(sentence["caption"])
            # verify
            assert all(len(v) == 20 for _, v in json_ref.items())
            self.json_ref = {k[len("video"):]: v for k, v in json_ref.items()}


    def __len__(self):
        return len(self.sentences)

    def _get_video(self, video_id):
        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=os.path.join(self.video_root, f"{video_id}.mp4"),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.mode == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)

        input_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        input_mask = torch.zeros(self.max_words, dtype=torch.long)
        input_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1

        video, video_mask = self._get_video(video_id)
        input_labels = torch.cat((input_ids[1:], torch.IntTensor([0])))
        return video, video_id, video_mask, input_ids, input_mask, input_labels, sentence

def collate_fn_caption_e2e(batch):
    video, video_id, video_mask, caption_ids, caption_mask, caption_labels, sentence = zip(*batch)
    '''2014/4/2注释7行'''
    bsz = len(video)
    video_feature = video[0]
    for key in video[0].keys():
        video_feature[key] = torch.cat([item[key][None, ...] for item in video], dim=0)
    # video_feature = torch.cat([item[None, ...] for item in video], dim=0)  # (bsz, sample_numb, dim_2d)
    video_mask = torch.cat([item[None, ...] for item in video_mask], dim=0)
    caption_ids = torch.cat([item[None, ...] for item in caption_ids], dim=0)
    caption_masks =torch.cat([item[None, ...] for item in caption_mask], dim=0)
    caption_labels = torch.cat([item[None, ...] for item in caption_labels], dim=0)

    captions = [item for item in sentence]

    video_id = list(video_id)

    return video_feature, video_mask.float(), caption_ids.long(), caption_labels.float(), caption_masks.float(), \
           captions, video_id