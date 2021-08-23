import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import glob
import datetime

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable
from tbpp_utils import PriorUtil

from pictText_utils import Generator
from data_pictText import InputGenerator

import argparse

data_path = "/home/sparsh.garg/nikhil/pictionary_redux/dataset/data_multiclass_circle"
save_path = "./saved_datasets"
num_classes = 2 # includes background class

parser = argparse.ArgumentParser(description='Hyperparameters')

parser.add_argument('--data', type=str, required=False, default=data_path)
parser.add_argument('--nc', type=int, required=False, default=num_classes)
parser.add_argument('--save', type=str, required=False, default=save_path)

# data-type
parser.add_argument('--RorA', type=str, required=False, default="real")
parser.add_argument('--className', type=str, required=False, default="text")

args = parser.parse_args()
print(args)

data_path = args.data
save_path = args.save
num_classes = args.nc # includes background class

augmented = args.RorA
class_name = args.className



gen = Generator(data_path, padding=0)
ds_train = gen.getDS("train", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, num_classes=num_classes, augmented=augmented, class_name=class_name)
ds_val = gen.getDS("val", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, num_classes=num_classes, augmented=augmented, class_name=class_name)
ds_test = gen.getDS("test", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, num_classes=num_classes, augmented=augmented, class_name=class_name)


if not os.path.exists(f"{save_path}/train"):
    os.makedirs(f"{save_path}/train")

if not os.path.exists(f"{save_path}/val"):
    os.makedirs(f"{save_path}/val")

if not os.path.exists(f"{save_path}/test"):
    os.makedirs(f"{save_path}/test")


def save_data(ds, path):

    color_map = ['red', 'blue', 'green', 'yellow']

    for i, sample in enumerate(ds):
        filename = str(i)
        try:
            print(sample)
            img, box = sample
            box, anomaly_classes = box
        except:
            fig = plt.figure(figsize=(10, 10))
            fig.savefig(f"{path}/{filename}.png")
            plt.close('all')
            continue

        fig = plt.figure(figsize=(10, 10))
        for j in range(len(box)):
            p = Polygon(list(box[j]), closed=True, edgecolor=color_map[anomaly_classes[j]-1], facecolor="none")
            ax = plt.gca()
            ax.add_patch(p)
        # ax.set_xlim(0,700)
        # ax.set_ylim(0,700)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        fig.savefig(f"{path}/{filename}.png")

        plt.close('all')

save_data(ds_train, f"{save_path}/train")
save_data(ds_val, f"{save_path}/val")
save_data(ds_test, f"{save_path}/test")
