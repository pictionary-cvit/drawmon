import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np

from tqdm import tqdm

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable


from pictText_utils import Generator
from data_pictText import InputGenerator, ImageInputGenerator, ImageInputGeneratorMulticlass

from tbpp_utils import PriorUtil

import tensorflow as tf

import argparse
import glob

from cocoevals import PycocoMetric

import matplotlib.pyplot as plt
import cv2

"""
Assumption:
Baseline: CanvasNet
activation: relu/mish
"""

parser = argparse.ArgumentParser(description='Hyperparameters')

parser.add_argument('--data', type=str, required=True, default='')
parser.add_argument('--bs', type=int, required=False, default=16)
parser.add_argument('--ct', type=float, required=False, default=0.4)
parser.add_argument('--split', type=str, choices=['val', 'test', 'train', "**"], required=False, default='val')
parser.add_argument('--nc', type=int, required=False, default=2)
parser.add_argument('--save', type=str, required=True, default='')

args = parser.parse_args()
print(args)

data_path = args.data
save_dir = args.save
confidence_threshold = args.ct
num_classes = args.nc
data_split = args.split
batch_size = args.bs

mirrored_strategy = tf.distribute.MirroredStrategy()

model = TBPP512_dense_separable(input_shape=(512, 512, 1), 
scale=0.9, 
isQuads=False, 
isRbb=False, 
num_dense_segs=3, 
use_prev_feature_map=False, 
num_multi_scale_maps=5, 
num_classes=num_classes)

prior_util = PriorUtil(model)
classes = ["bg", "text", "number", "symbol", "circle"]

os.makedirs(save_dir, exist_ok=True)

def renderPreds(imgs, truths=None):   
    rends = []
    for i in range(imgs.shape[0]):
        fig = plt.figure(figsize=[9]*2)
        im = np.pad(np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])), pad_width=(15), constant_values=(0))
        print(im.shape)
        plt.imshow(1-im, cmap='gray')
        
        res_truth = truths[i]    
        prior_util.plot_results([], gt_data_decoded=res_truth, show_labels=False, classes=classes, hw = (imgs[i].shape[0], imgs[i].shape[1]), pad=15)

        plt.axis('off')
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')

    return np.array(rends)

if num_classes == 2:
    gen_train = ImageInputGenerator(data_path, batch_size, "train", give_idx=True)
    gen_val = ImageInputGenerator(data_path, batch_size, "val", give_idx=True)
else:
    gen_train = ImageInputGeneratorMulticlass(
        data_path, batch_size, "train", give_idx=False
    )
    gen_val = ImageInputGenerator(data_path, batch_size, "val", give_idx=False)

dataset_train, dataset_val = gen_train.get_dataset(), gen_val.get_dataset()

dist_dataset_train = mirrored_strategy.experimental_distribute_dataset(dataset_train)
dist_dataset_val = mirrored_strategy.experimental_distribute_dataset(dataset_val)

sample_count = 0
for dist_inputs in dist_dataset_train:
    save_prob = random.uniform(0, 1)
    if save_prob < 0.5:    
        x, y_true = dist_inputs
        x_list = mirrored_strategy.experimental_local_results(x)
        y_true_list = mirrored_strategy.experimental_local_results(y_true)
        # idx_list = mirrored_strategy.experimental_local_results(idx)
        assert len(x_list) == len(y_true_list)
        for i, x_i in enumerate(x_list):
            y_true_i = y_true_list[i]
            rends = renderPreds(x_i, y_true_i)

            # save rends
            for ii in range(len(rends)):
                filenameB = f"{save_dir}/{sample_count}.png"
                cv2.imwrite(filenameB, rends[ii])
                sample_count += 1
