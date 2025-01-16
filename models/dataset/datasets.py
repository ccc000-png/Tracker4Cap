import json
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from models.layers.clip import clip

class CaptionDataset(Dataset):
    def __init__(self, cfgs, mode):
        """1.get id"""
        self.mode = mode
        self.dataset_name = cfgs.data.dataset
        videos_split = cfgs.data.videos_split.format(mode)
        with open(videos_split, 'rb') as f:
            video_ids = pickle.load(f)
        self.video_ids = video_ids
        # 获取训练、测试ids
        """1.get video"""
        self.visual_path = cfgs.data.visual_features
        self.objects_visual_path = cfgs.data.object_features.format(mode)
        sample_numb = cfgs.sample_numb

        self.visual_dict = {}
        self.objects_dict = {}
        # visual dict
        for vid in tqdm(self.video_ids):
            temp_feat = np.load(os.path.join(self.visual_path, vid + '.npy'))

            sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
            self.visual_dict[vid] = temp_feat[sampled_idxs]

        """2.get ann"""
        self.max_words=cfgs.decoder.max_caption_len
        # msvd\msrvtt提取方式不同
        self.ann = json.load(open(cfgs.data.ann_root, 'r'))
        self.sentences = []
        json_ref = {k: [] for k in self.video_ids}
        vid2sentence = defaultdict(list)
        if cfgs.data.dataset == 'msvd':
            for item in tqdm(self.ann["metadata"]):
                if item["video_id"] in self.video_ids:
                    if mode == "train":
                        self.sentences.append([item["video_id"], [item["sentence"]]])
                    else:
                        vid2sentence[item["video_id"]].append(item["sentence"])
                        self.sentences = list(vid2sentence.items())
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item["sentence"])
        elif cfgs.data.dataset == 'msrvtt':
            for item in tqdm(self.ann['sentences']):
                if item["video_id"] in self.video_ids:
                    if mode == "train":
                        self.sentences.append([item["video_id"], [item['caption']]])
                    else:
                        vid2sentence[item["video_id"]].append(item['caption'])
                        self.sentences = list(vid2sentence.items())
                if item["video_id"] in json_ref:
                    json_ref[item["video_id"]].append(item['caption'])
        self.json_ref = json_ref


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)
        # sentence = sentence_list[0]
        captions = sentence
        feature2d = self.visual_dict[video_id]


        caption_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        caption_mask = torch.zeros(self.max_words, dtype=torch.long)
        caption_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1
        caption_labels = torch.cat((caption_ids[1:], torch.IntTensor([0])))

        return torch.FloatTensor(feature2d),\
               caption_ids, caption_mask, caption_labels, captions, video_id

def collate_fn_caption(batch):
    feature2ds, caption_ids, caption_mask, caption_labels, captions, video_id = zip(*batch)

    feature2ds = torch.cat([item[None, ...] for item in feature2ds], dim=0)  # (bsz, sample_numb, dim_2d)
    caption_ids = torch.cat([item[None, ...] for item in caption_ids], dim=0)
    caption_masks =torch.cat([item[None, ...] for item in caption_mask], dim=0)
    caption_labels = torch.cat([item[None, ...] for item in caption_labels], dim=0)

    captions = [item for item in captions]

    video_id = list(video_id)

    return feature2ds.float(), caption_ids.long(),  caption_labels.float(), caption_masks.float(),captions, video_id

