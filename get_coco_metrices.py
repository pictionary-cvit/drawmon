import numpy as np
import os
from tensorflow.python.framework.tensor_util import constant_value
from tqdm import tqdm

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable


from pictText_utils import Generator
from data_pictText import InputGenerator, ImageInputGenerator

from tbpp_utils import PriorUtil

import tensorflow as tf

import argparse
import sys
import glob

from cocoevals import PycocoMetric

import matplotlib.pyplot as plt
import cv2

# Get Argument
data_path = "/home/nikhil.bansal/pictionary_redux/pictionary_redux/dataset/obj_detection_data"
batch_size = 16
confidence_threshold = 0.4
scale = 0.9
isQuads = 'False'
isRbb='False'

num_dense_segs=3 # default = 3
use_prev_feature_map='False' # default = False
num_multi_scale_maps=5 # default = 5
num_classes=1 + 1 # default = 2
model_name = "ds"
weights_path = None
data_split='val'

onlyLastwt='False'
activation='relu'

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data', type=str, required=False, default=data_path)
parser.add_argument('--bs', type=int, required=False, default=batch_size)
parser.add_argument('--ct', type=float, required=False, default=confidence_threshold)
parser.add_argument('--scale', type=float, required=False, default=scale)
parser.add_argument('--isQ', type=eval, 
                      choices=[True, False], required=False, default=isQuads)
parser.add_argument('--isR', type=eval, 
                      choices=[True, False], required=False, default=isRbb)
parser.add_argument('--nds', type=int, required=False, default=num_dense_segs)
parser.add_argument('--isPMap', type=eval, 
                      choices=[True, False], required=False, default=use_prev_feature_map)
parser.add_argument('--nmsm', type=int, required=False, default=num_multi_scale_maps)
parser.add_argument('--nc', type=int, required=False, default=num_classes)
parser.add_argument('--model', type=str, choices=['tbpp', 'ds', 'dsod'], required=False, default=model_name)
parser.add_argument('--split', type=str, choices=['val', 'test', 'train', "**"], required=False, default=data_split)
parser.add_argument('--wpath', type=str, required=True, default=None)

parser.add_argument('--onlyLastwt', type=eval, choices=[True, False], required=False, default=onlyLastwt)
parser.add_argument('--activation', type=str, required=False, default='relu')
parser.add_argument('--onlyWt', type=int, required=False, default=None)


args = parser.parse_args()
print(args)

data_path = args.data


batch_size = args.bs
confidence_threshold = args.ct
scale = args.scale
isQuads = args.isQ
isRbb=args.isR

num_dense_segs=args.nds # default = 3
use_prev_feature_map=args.isPMap # default = True
num_multi_scale_maps=args.nmsm # default = 5
activation = args.activation
num_classes=args.nc # default = 1 + 1 (bg + text)
weights_path=args.wpath
data_split=args.split

only_last = args.onlyLastwt

onlyWt = args.onlyWt

model_name=args.model

if model_name == 'ds':
    model = TBPP512_dense_separable(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_dense_segs=num_dense_segs, use_prev_feature_map=use_prev_feature_map, num_multi_scale_maps=num_multi_scale_maps, num_classes=num_classes, activation=activation)
elif model_name == 'dsod':
    model = TBPP512_dense(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes)
elif model_name == 'tbpp':
    model = TBPP512(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes)
else:
    sys.exit('Model Not Supported\nChoices: [ds, dsod]')


prior_util = PriorUtil(model)
priors_xy = tf.Variable(prior_util.priors_xy/prior_util.image_size, dtype=tf.float32)
priors_wh = tf.Variable(prior_util.priors_wh/prior_util.image_size, dtype=tf.float32)
priors_variances = tf.Variable(prior_util.priors_variances, dtype=tf.float32)


gen_val = ImageInputGenerator(data_path, batch_size, data_split)

dataset_val = gen_val.get_dataset()
print(f"Number of validation batches: {len(dataset_val)}")



pycoco_metric = PycocoMetric(iou_threshold = 0.5,
                             confidence_threshold = 0.3, 
                             top_k = 200, num_classes=num_classes)

def get_AP(y_true, y_pred):
    coco_metrics = pycoco_metric(y_true, y_pred)
    return coco_metrics


y_true = []
y_pred = []


classes = ["bg", "text", "number", "symbol", "circle"]
checkdir = f'{weights_path}/cocometrics/{data_split}'

if weights_path[-1] != '/':
    weight_files = glob.glob(f'{weights_path}/*.h5')
else:
    weight_files = glob.glob(f'{weights_path}*.h5')
weight_files.sort()

if only_last == True:
    weight_files = [weight_files[-1]]


if onlyWt is not None:
    weight_files = [weight_files[onlyWt-1]]

def renderPreds(imgs, preds, truths=None, only_img = False):   
    rends = []
    for i in range(imgs.shape[0]):
        fig = plt.figure(figsize=[9]*2)
        im = np.pad(np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])), pad_width=(15), constant_values=(0))
        print(im.shape)
        plt.imshow(1-im, cmap='gray')
        
        if not only_img:
            res = preds[i]
            res_truth = truths[i]
            # prior_util.plot_gt()
            # TextWordId, MultiNumberId, SymbolId, CircleId
            # prior_util.plot_results(res, gt_data_decoded=res_truth, show_labels=True, classes=classes)
            prior_util.plot_results(res, show_labels=False, classes=classes, hw = (imgs[i].shape[0], imgs[i].shape[1]), pad=15)

        plt.axis('off')
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')
    
    return np.array(rends)

def lprint(s, class_index):
    if class_index == -1: class_index = "all"
    else: class_index = str(class_index)
    with open(f"{weights_path}/results/{data_split}/result_log_{class_index}.txt", "a") as fil:
        fil.write(s)
        fil.write('\n')
    

def evaluate(class_idx = -1):
    if class_idx == -1:
        class_name = "all"
    else: 
        class_name = classes[class_idx]

    os.makedirs(f"{checkdir}/tb/{class_idx}", exist_ok=True)
    val_summary_writer = tf.summary.create_file_writer(f"{checkdir}/tb/{class_idx}")
    best_metrics=None
    best_epoch=0
    

    for filep in tqdm(range(len(weight_files))):
        # print(weight_files)
        filepath = weight_files[filep]
        model.load_weights(filepath)
        
        imgs = []
        y_true = []
        y_pred = []

        isAnyPredictions = False
            
        for ii, (images, data) in enumerate(dataset_val):
            # print(f"Input Size: {images.shape}")
            # print(f"Output Size: {data.shape}")
            preds = model.predict_on_batch(images)
            # print(f"Prediction Size: {preds.shape}")
            for i in range(len(preds)):
                res = prior_util.decode(preds[i], class_idx = class_idx, confidence_threshold = confidence_threshold, fast_nms=False)
                truths = prior_util.decode(data[i], class_idx = class_idx, confidence_threshold = confidence_threshold, fast_nms=False)
                    
                imgs.append(images[i])
                y_true.append(truths)
                if (res.shape[0] != 0): isAnyPredictions = True
                y_pred.append(res)
        
        if (isAnyPredictions == False):
            # No predictions for the whole val-set
            print(f"Skipping epoch-{filep+1}, as there are no predictions in the whole val-set.")
        else:    
            with val_summary_writer.as_default():
                metrics = get_AP(np.array(y_true), np.array(y_pred))
                # Current AP@0.50 is given by metrics[1][1]
                for metric, metric_value in metrics:
                    tf.summary.scalar(str(metric), metric_value, step=filep+1)
            
                if best_metrics == None:
                    best_metrics = metrics
                    best_epoch = filep + 1
                else:
                    if (metrics[1][1] > best_metrics[1][1]):
                        print(f"Best epoch changed from {best_epoch} to {filep+1}")
                        best_metrics = metrics
                        best_epoch = filep + 1
    
    def plot_best_epoch():
        """Plot and save predictions from model having weights that gives best AP"""
        print(f"Plot epoch: {best_epoch}")
        idx = best_epoch - 1
        filepath = weight_files[idx]
        model.load_weights(filepath)
        
        sample_count = 0
        detection_accuracy = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
    
        for ii, (images, data) in enumerate(dataset_val):
            preds = model.predict_on_batch(images)
            for i in range(len(preds)):
                res = prior_util.decode(preds[i], class_idx = class_idx, confidence_threshold = confidence_threshold, fast_nms=False)
                truths = prior_util.decode(data[i], class_idx = class_idx, confidence_threshold = confidence_threshold, fast_nms=False)
                
                if (len(truths) == 0 and len(res) == 0) or (len(truths) != 0 and len(res) != 0):
                    detection_accuracy += 1
                if (len(truths) == 0 and len(res) == 0):
                    tn += 1
                if (len(truths) == 0 and len(res) != 0):
                    fp += 1
                if (len(truths) != 0 and len(res) == 0):
                    fn += 1
                if (len(truths) != 0 and len(res) != 0):
                    tp += 1

                render = renderPreds(np.array([images[i]]), np.array([res]), np.array([truths]))
                dirpath = f"{weights_path}/results/{data_split}/{weight_files[filep].split('/')[-1]}_{class_idx}"

                os.makedirs(dirpath, exist_ok=True)
                filename = f"{dirpath}/{sample_count}.png"
                sample_count += 1
                cv2.imwrite(filename, render[0])
        
    
    
        def f1(p,r):
            return 2*p*r/(p+r) if p+r != 0 else 0

        p = tp/(tp + fp)
        r = tp/(tp + fn)
        acc = detection_accuracy/sample_count

        lprint(f'class {class_name}: Detection: acc = {acc}, p = {p}, r = {r}, f1 = {f1(p, r)}', class_idx)
        lprint(f'class {class_name}: Detection: acc = {round(acc, 2)}, p = {round(p, 2)}, r = {round(r, 2)}, f1 = {round(f1(p,r), 2)}', class_idx)
        lprint(f'class {class_name}: Best Epoch: {best_epoch}', class_idx)
        lprint(f'class {class_name}: Best Metrics: {best_metrics}', class_idx)

        ap = best_metrics[1][1]
        ar = best_metrics[7][1]

        lprint(f'class {class_name}: ap = {ap}, ar = {ar}, f1 = {f1(ap,ar)}', class_idx)
        lprint(f'class {class_name}: ap = {round(ap, 2)}, ar = {round(ar, 2)}, f1 = {round(f1(ap,ar), 2)} ', class_idx)
    
    plot_best_epoch()

#for class_idx in range(1, num_classes, 1):
#    evaluate(class_idx)
evaluate(-1)

