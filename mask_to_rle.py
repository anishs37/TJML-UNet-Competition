from PIL import Image
import os
from tqdm import tqdm
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import ast

def mask2rle(pred):
    shape = pred.shape
    mask = np.zeros([shape[0], shape[1]], dtype=np.uint8)
    points = np.where(pred == 1)
    if len(points[0]) > 0:
        mask[points[0], points[1]] = 1
        mask = mask.reshape(-1, order='F')
        pixels = np.concatenate([[0], mask, [0]])
        rle = np.where(pixels[1:] != pixels[:-1])[0]
        rle[1::2] -= rle[::2]
    else:
        return ''
    return ' '.join(str(r) for r in rle)
