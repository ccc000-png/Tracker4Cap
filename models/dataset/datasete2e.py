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

from models.dataset.utils import read_frames_decord
from models.layers.clip import clip


class CaptionDatasetE2E(Dataset):
    def __init__(self, cfgs, mode,transform):
        """1.获取文本信息"""
        self.mode = mode

        self.dataset_name = cfgs.data.dataset
        self.data_root = cfgs.data.data_root
        self.video_root = cfgs.data.video_root
        self.sample_numb = cfgs.sample_numb
        self.transform=transform
        self.max_words = cfgs.decoder.max_caption_len
        self.ann = json.load(open(cfgs.data.ann_root, 'r'))
        self.sentences = []

        vid2sentence = defaultdict(list)
        if cfgs.data.dataset == 'msvd':
            self.video_ids = self.ann[mode].copy()
            json_ref = {k: [] for k in self.video_ids}
            for item in tqdm(self.ann["metadata"]):
                if item["video_id"] in self.video_ids:
                    vid2sentence[item["video_id"]].append(item["sentence"])
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item["sentence"])
        elif cfgs.data.dataset == 'msrvtt':
            video_ids = [self.ann['videos'][idx]['video_id'] for idx in range(len(self.ann['videos']))]
            all_split_video_ids = {"train": video_ids[:6513], "valid": video_ids[6513:6513 + 497],
                                   "test": video_ids[6513 + 497:]}
            self.video_ids = all_split_video_ids[mode].copy()
            json_ref = {k: [] for k in self.video_ids}
            for item in tqdm(self.ann['sentences']):
                if item["video_id"] in self.video_ids:
                    vid2sentence[item["video_id"]].append(item['caption'])
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item['caption'])
        self.sentences = list(vid2sentence.items())
        self.json_ref = json_ref

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)
        # sentence = sentence_list[0]

        caption_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        caption_mask = torch.zeros(self.max_words, dtype=torch.long)
        caption_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1
        caption_labels = torch.cat((caption_ids[1:], torch.IntTensor([0])))
        if self.dataset_name=='msvd':
            video_path = os.path.join(self.video_root, video_id + '.avi')
        elif self.dataset_name=='msrvtt':
            video_path = os.path.join(self.video_root, video_id + '.mp4')

        while True:
            try:
                frames = read_frames_decord(video_path, self.data_root, video_id, sample_frames=self.sample_numb, sample='rand')# (T, C, H, W)

            except:
                time.sleep(0.01)
                video_path = os.path.join(self.video_root, video_id + '.mp4')
                continue
            break
        frames = [self.transform(PIL.Image.open(frame)) for frame in frames]
        video = torch.tensor(np.stack(frames)) # (T, H, W, C)
        # video=frames
        return video, video_id, caption_ids, caption_mask, caption_labels, sentence

def collate_fn_caption_e2e(batch):
    video, video_id, caption_ids, caption_mask, caption_labels, sentence = zip(*batch)
    '''2014/4/2注释7行'''
    bsz = len(video)

    # video_feature = [item for item in video]
    video_feature = torch.cat([item[None, ...] for item in video], dim=0)  # (bsz, sample_numb, dim_2d)
    caption_ids = torch.cat([item[None, ...] for item in caption_ids], dim=0)
    caption_masks =torch.cat([item[None, ...] for item in caption_mask], dim=0)
    caption_labels = torch.cat([item[None, ...] for item in caption_labels], dim=0)

    captions = [item for item in sentence]

    video_id = list(video_id)

    return video_feature.float(), caption_ids.long(),  caption_labels.float(), caption_masks.float(),captions, video_id