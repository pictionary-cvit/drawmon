import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import os
from tensorflow.python.framework.tensor_util import constant_value
from tqdm import tqdm

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable

from pictText_utils import Generator
from data_pictText import InputGenerator

from tbpp_utils import PriorUtil

import tensorflow as tf

import argparse
import sys
import glob

from cocoevals import PycocoMetric

import matplotlib.pyplot as plt
import cv2
# Get Argument

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data', type=str, required=True, default='')
parser.add_argument('--save', type=str, required=True, default='')
parser.add_argument('--split', type=str, required=True, default='')
parser.add_argument('--nc', type=int, required=True, default=5)

args = parser.parse_args()
print(args)

data_path = args.data
save_path = args.save
data_split = args.split
num_classes = args.nc

if num_classes > 2:
    output_path = f"{save_path}/{data_split}_multiclass"
else:
    output_path = f"{save_path}/{data_split}_textOnly"
names_list_path = f"{data_split}.txt"
os.makedirs(output_path, exist_ok=True)

gen = Generator(data_path, padding=0)
ds_val = gen.getDS(data_split, stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, augmented="**", num_classes=num_classes)

batch_size = 5
gen_val = InputGenerator(ds_val, None, batch_size, 5, encode=False, overlap_threshold=0.5, split=data_split, 
    encode=False,
    num_classes=num_classes, 
    isFlattenBoxes=False, 
    isConvertToCentroid=True)

img_count = 0
ds_iterator = gen_val.get_dataset()
for imgs, targets in ds_iterator:
    for i in range(imgs.shape[0]):
        try:
            img = imgs[i]
            labels = targets[i][:, -1]
            boxes = targets[i][:, 0:-1]

            filename = f"{img_count}_{data_split}"
            with open(os.path.join(output_path, filename + ".txt"), "w") as fil:
                for i in range(len(labels)):
                    fbox = [int(labels[i])-1]
                    for ii in range(4):
                        fbox.append(boxes[i][ii])

                    fbox = list(map(str, fbox))
                    fil.write(' '.join(fbox))
                    fil.write("\n")

            cv2.imwrite(os.path.join(output_path, filename + ".jpg"), img)

            with open(os.path.join(output_path, names_list_path), "a") as fil:
                fil.write(os.path.join("data/obj/", filename+".jpg"))
                fil.write("\n")

            img_count += 1
        
        except:
            continue
