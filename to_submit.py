import argparse
from pathlib import Path
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as im
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from predict import predict_img, mask_to_image
from mask_to_rle import mask2rle
import csv

header = ['Id', 'Mask']
ids = np.load('./id.npy')
imgs = np.load('./img.npy')

with open('submission13.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    counter = 0
    for id_s in ids:
        data = []
        data.append(id_s)

        img_counter = imgs[counter]

        net = UNet(n_channels=3, n_classes=2)

        d_use = torch.device('cpu')

        net.to(device=d_use)
        net.load_state_dict(torch.load("./checkpoints/checkpoint_epoch20.pth", map_location=d_use))

        img = img_counter
        pil_img = im.fromarray(img)

        mask_1 = predict_img(net=net, full_img=pil_img, scale_factor=1, out_threshold=0.5, device=d_use)
        mask_1 = np.argmax(mask_1, axis=0)
        pred = mask_1
        str_to_add = mask2rle(pred)
        data.append(str_to_add)

        writer.writerow(data)
        counter = counter + 1
        print(counter)