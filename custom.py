import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, masks, scale: float = 1.0, mask_suffix: str = ''):
        self.images = images
        self.masks = (masks > 0)*255 
        self.mask_suffix = mask_suffix
        assert(len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        
        img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, idx):
        img = self.preprocess(self.images[idx], is_mask=False)
        mask = self.preprocess(self.masks[idx], is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
