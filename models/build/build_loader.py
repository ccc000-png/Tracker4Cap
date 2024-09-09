import torch
from torch.utils.data import DataLoader
from models.dataset.datasete2e import CaptionDatasetE2E, collate_fn_caption_e2e
from models.dataset.datasets import CaptionDataset, collate_fn_caption

def build_dataset(cfgs,transform=None):
    if cfgs.data.train_type == 'preprocess':
        train_dataset = CaptionDataset(cfgs=cfgs, mode='train')
        valid_dataset = CaptionDataset(cfgs=cfgs, mode='valid')
        test_dataset = CaptionDataset(cfgs=cfgs, mode='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, shuffle=False,
                             collate_fn=collate_fn_caption, num_workers=0)

    else:
        train_dataset = CaptionDatasetE2E(cfgs=cfgs, mode='train',transform=transform)
        valid_dataset = CaptionDatasetE2E(cfgs=cfgs, mode='valid',transform=transform)
        test_dataset = CaptionDatasetE2E(cfgs=cfgs, mode='test',transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, shuffle=True,
                                  collate_fn=collate_fn_caption_e2e, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, shuffle=True,
                                  collate_fn=collate_fn_caption_e2e, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, shuffle=False,
                                 collate_fn=collate_fn_caption_e2e, num_workers=0)
    return train_loader, valid_loader,test_loader
