"""
Purpose:
    1. Generate GT (labeled(i.e box drawn on img), unlabeled) images
    2. Corresponding to each GT above, generate Preds labeled images 
    3. For each GT unlabeled image, generate list of (gtBoxes, predBoxes)
"""

import sys
import os
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

# Model-A
parser.add_argument('--wpath', type=str, required=True, default=None)
parser.add_argument('--activation', type=str, required=False, default='mish')

args = parser.parse_args()
print(args)

data_path = args.data
save_dir = args.save

activation = args.activation

weight_path = args.wpath

confidence_threshold = args.ct
num_classes = args.nc
data_split = args.split
batch_size = args.bs

model = TBPP512_dense_separable(
    input_shape=(512, 512, 1), 
    softmax=True, 
    scale=0.9, 
    isQuads=False, 
    isRbb=False, 
    num_dense_segs=3, 
    use_prev_feature_map=False, 
    num_multi_scale_maps=5, 
    num_classes=num_classes, 
    activation=activation
)

prior_util = PriorUtil(model)

if num_classes > 2 and data_split == 'train':
    gen_val = ImageInputGeneratorMulticlass(data_path, batch_size, data_split, give_idx=False)
else:
    gen_val = ImageInputGenerator(data_path, batch_size, data_split, give_idx=False)

dataset_val = gen_val.get_dataset()

classes = ["bg", "text", "number", "symbol", "circle"]

def renderPreds(imgs, preds, prior_util, truths=None, only_img = False, pad_width=15):   
    rends = []
    for i in range(imgs.shape[0]):
        fig = plt.figure(figsize=[9]*2)
        im = np.pad(np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])), pad_width=(pad_width), constant_values=(0))
        print(im.shape)
        plt.imshow(1-im, cmap='gray')
        
        if not only_img:
            res = preds[i]
            res_truth = truths[i]
            
            prior_util.plot_results(res, gt_data_decoded=res_truth, show_labels=False, classes=classes, hw = (imgs[i].shape[0], imgs[i].shape[1]), pad=pad_width)

        plt.axis('off')
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')

    return np.array(rends)

# For storing model (<session>.png, <session>_gts.png, <session>_preds.png)
save_i = f"{save_dir}/{data_split}/imgs"
save_g = f"{save_dir}/{data_split}/gts"
save_p = f"{save_dir}/{data_split}/preds"
os.makedirs(save_i, exist_ok=True)
os.makedirs(save_g, exist_ok=True)
os.makedirs(save_p, exist_ok=True)


model.load_weights(weight_path)

sample_count = 0
all_truths = []
all_preds = []
for ii, (images, data) in enumerate(dataset_val):
    preds = model.predict_on_batch(images)
    
    for i in range(len(images)):
        res = prior_util.decode(preds[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        truths = prior_util.decode(data[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
               
        # save img
        render = renderPreds(np.array([images[i]]), None, prior_util, None, True, 0)
        filename = f"{save_i}/{sample_count}.png" # save img
        cv2.imwrite(filename, render[0])

        # save model gts
        render = renderPreds(np.array([images[i]]), np.array([[]]), prior_util, np.array([truths]))
        filename = f"{save_g}/{sample_count}.png" # save gt
        cv2.imwrite(filename, render[0])

        # save model preds
        render = renderPreds(np.array([images[i]]), np.array([res]), prior_util, np.array([[]]))
        filename = f"{save_p}/{sample_count}.png" # save preds
        cv2.imwrite(filename, render[0])

        all_truths.append(np.array(truths))
        all_preds.append(np.array(res))

        sample_count += 1


np.save(f"{save_dir}/{data_split}/truth.npy", np.array(all_truths))
np.save(f"{save_dir}/{data_split}/preds.npy", np.array(all_preds))