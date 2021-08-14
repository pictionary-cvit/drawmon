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
parser.add_argument('--wpathA', type=str, required=True, default=None)
parser.add_argument('--activationA', type=str, required=False, default='mish')

# Model-B
parser.add_argument('--wpathB', type=str, required=True, default=None)
parser.add_argument('--activationB', type=str, required=False, default='relu')

args = parser.parse_args()
print(args)

data_path = args.data
save_dir = args.save

activationA = args.activationA
activationB = args.activationB

weight_pathA = args.wpathA
weight_pathB = args.wpathB

confidence_threshold = args.ct
num_classes = args.nc
data_split = args.split
batch_size = args.bs

modelA = TBPP512_dense_separable(
    input_shape=(512, 512, 1), 
    softmax=True, 
    scale=0.9, 
    isQuads=False, 
    isRbb=False, 
    num_dense_segs=3, 
    use_prev_feature_map=False, 
    num_multi_scale_maps=5, 
    num_classes=num_classes, 
    activation=activationA
)

modelB = TBPP512_dense_separable(
    input_shape=(512, 512, 1), 
    softmax=True, 
    scale=0.9, 
    isQuads=False, 
    isRbb=False, 
    num_dense_segs=3, 
    use_prev_feature_map=False, 
    num_multi_scale_maps=5, 
    num_classes=num_classes, 
    activation=activationB
)

prior_utilA = PriorUtil(modelA)
prior_utilB = PriorUtil(modelB)


if num_classes > 2 and data_split == 'train':
    gen_val = ImageInputGeneratorMulticlass(data_path, batch_size, data_split, give_idx=False)
else:
    gen_val = ImageInputGenerator(data_path, batch_size, data_split, give_idx=False)

dataset_val = gen_val.get_dataset()

classes = ["bg", "text", "number", "symbol", "circle"]

def renderPreds(imgs, preds, prior_util, truths=None, only_img = False):   
    rends = []
    for i in range(imgs.shape[0]):
        fig = plt.figure(figsize=[9]*2)
        im = np.pad(np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])), pad_width=(15), constant_values=(0))
        print(im.shape)
        plt.imshow(1-im, cmap='gray')
        
        if not only_img:
            res = preds[i]
            res_truth = truths[i]
            
            prior_util.plot_results(res, gt_data_decoded=res_truth, show_labels=False, classes=classes, hw = (imgs[i].shape[0], imgs[i].shape[1]), pad=15)

        plt.axis('off')
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')

    return np.array(rends)

# For storing Model-A predictions
saveA = f"{save_dir}/{data_split}/model_{activationA}"
os.makedirs(saveA, exist_ok=True)

# For storing Model-B predicitons
saveB = f"{save_dir}/{data_split}/model_{activationB}"
os.makedirs(saveB, exist_ok=True)

# For storing results where number of predictions of Model-A are greater
saveAB = f"{save_dir}/{data_split}/{activationA}_{activationB}"
os.makedirs(saveAB, exist_ok=True)
os.makedirs(f"{saveAB}/A", exist_ok=True)
os.makedirs(f"{saveAB}/B", exist_ok=True)

# For storing results where number of predictions of Model-B are greater
saveBA = f"{save_dir}/{data_split}/{activationB}_{activationA}"
os.makedirs(saveBA, exist_ok=True)
os.makedirs(f"{saveBA}/A", exist_ok=True)
os.makedirs(f"{saveBA}/B", exist_ok=True)


modelA.load_weights(weight_pathA)
modelB.load_weights(weight_pathB)

sample_count = 0

for ii, (images, data) in enumerate(dataset_val):
    predsA = modelA.predict_on_batch(images)
    predsB = modelB.predict_on_batch(images)
    
    for i in range(len(images)):
        resA = prior_utilA.decode(predsA[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        truthsA = prior_utilA.decode(data[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        
        resB = prior_utilB.decode(predsB[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        truthsB = prior_utilB.decode(data[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        
        # save model-A predictions
        renderA = renderPreds(np.array([images[i]]), np.array([resA]), prior_utilA, np.array([truthsA]))
        filenameA = f"{saveA}/{sample_count}.png"
        cv2.imwrite(filenameA, renderA[0])

        # save model-B predictions
        renderB = renderPreds(np.array([images[i]]), np.array([resB]), prior_utilB, np.array([truthsB]))
        filenameB = f"{saveB}/{sample_count}.png"
        cv2.imwrite(filenameB, renderB[0])

        if len(resA) > len(resB):
            cv2.imwrite(f"{saveAB}/A/{sample_count}.png", renderA[0])    
            cv2.imwrite(f"{saveAB}/B/{sample_count}.png", renderB[0])    
        elif len(resA) < len(resB):
            cv2.imwrite(f"{saveBA}/A/{sample_count}.png", renderA[0])    
            cv2.imwrite(f"{saveBA}/B/{sample_count}.png", renderB[0])
        
        sample_count += 1
