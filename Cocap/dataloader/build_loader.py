import torch
from torch.utils.data import DataLoader
from Cocap.dataloader.datasete2e import CaptionDatasetE2E, collate_fn_caption_e2e

def build_dataset(cfgs,transform=None):
    train_dataset = CaptionDatasetE2E(cfgs=cfgs, mode='train',transform=transform)
    test_dataset = CaptionDatasetE2E(cfgs=cfgs, mode='test',transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.data.LOADER.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_caption_e2e, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.data.LOADER.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_caption_e2e, num_workers=0)
    return train_loader, test_loader
