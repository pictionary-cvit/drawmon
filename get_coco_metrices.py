import numpy as np
import os
from tqdm import tqdm

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable
from ssd_data import InputGenerator


from pictText_utils import Generator
from data_pictText import InputGenerator

from tbpp_utils import PriorUtil

import tensorflow as tf

import argparse
import sys
import glob

from cocoevals import PycocoMetric

# Get Argument
data_path = "/home/nikhil.bansal/pictionary_redux/pictionary_redux/dataset/obj_detection_data"
batch_size = 6
confidence_threshold = 0.4
scale = 0.9
isQuads = 'False'
isRbb='False'

num_dense_segs=3 # default = 3
use_prev_feature_map='True' # default = True
num_multi_scale_maps=5 # default = 5
num_classes=1 + 1 # default = 2
model_name = "ds"
weights_path = None
data_split='val'

onlyLastwt='False'


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
parser.add_argument('--model', type=str, choices=['std', 'ds', 'dsod'], required=False, default=model_name)
parser.add_argument('--split', type=str, choices=['val', 'test'], required=False, default=data_split)
parser.add_argument('--wpath', type=str, required=True, default=None)

parser.add_argument('--onlyLastwt', type=eval, required=False, default=onlyLastwt)


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
num_classes=args.nc # default = 1 + 1 (bg + text)
weights_path=args.wpath
data_split=args.split

only_last = args.onlyLastwt

model_name=args.model

if model_name == 'ds':
    model = TBPP512_dense_separable(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_dense_segs=num_dense_segs, use_prev_feature_map=use_prev_feature_map, num_multi_scale_maps=num_multi_scale_maps, num_classes=num_classes)
elif model_name == 'dsod':
    model = TBPP512_dense(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes)
elif model_name == 'std':
    model = TBPP512(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes)
else:
    sys.exit('Model Not Supported\nChoices: [ds, dsod]')

gen = Generator(data_path, padding=0)
ds_val = gen.getDS(data_split, augmented="**", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=2)

prior_util = PriorUtil(model)
priors_xy = tf.Variable(prior_util.priors_xy/prior_util.image_size, dtype=tf.float32)
priors_wh = tf.Variable(prior_util.priors_wh/prior_util.image_size, dtype=tf.float32)
priors_variances = tf.Variable(prior_util.priors_variances, dtype=tf.float32)


gen_val = InputGenerator(ds_val, prior_util, batch_size, 2, encode=True, overlap_threshold=0.5)
# gen_val_decoded = InputGenerator(ds_val, prior_util, batch_size, 2, encode=False)

iterator_val = gen_val.get_dataset()
# iterator_val_decoded = gen_val_decoded.get_dataset()
# print(len(iterator_val_decoded))
print(f"Number of validation batches: {len(iterator_val)}")



pycoco_metric = PycocoMetric(iou_threshold = 0.5,
                             confidence_threshold = 0.3, 
                             top_k = 200)

def get_AP(y_true, y_pred):
    coco_metrics = pycoco_metric(y_true, y_pred)
    return coco_metrics


y_true = []
y_pred = []



checkdir = f'{weights_path}/cocometrics'
val_summary_writer = tf.summary.create_file_writer(checkdir)

if weights_path[-1] != '/':
    weight_files = glob.glob(f'{weights_path}/*.h5')
else:
    weight_files = glob.glob(f'{weights_path}*.h5')
weight_files.sort()
best_metrics=None
best_epoch=0

if only_last == True:
    weight_files = [weight_files[-1]]
elif type(only_last) is not bool:
    weight_files = [weight_files[only_last-1]]
    #weight_files = weight_files[only_last-1:]

for fp in tqdm(range(len(weight_files))):
    filepath = weight_files[fp]
    model.load_weights(filepath)

    for ii, (images, data) in enumerate(iterator_val):
        preds = model.predict(images, batch_size=batch_size, verbose=1)        
        for i in range(len(preds)):
            res = prior_util.decode(preds[i], confidence_threshold, fast_nms=False)
            truths = prior_util.decode(data[i], confidence_threshold, fast_nms=False)

            y_true.append(truths)
            y_pred.append(res)
            
    with val_summary_writer.as_default():
    #if True:    
        metrics = get_AP(y_true, y_pred)
        label = 'Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ]'
        cur_label_value = None
        tf.summary.scalar('mAP@0.5', metrics[1][1], step=fp+1)
        tf.summary.scalar('mAR@0.5', metrics[7][1], step=fp+1)
        tf.summary.scalar('F1', 2*(metrics[1][1])*(metrics[7][1])/(metrics[1][1]+metrics[7][1]), step=fp+1)
        #print()
        for metric, metric_value in metrics:
            if (str(metric) == label):
                cur_label_value = metric_value
            tf.summary.scalar(str(metric), metric_value, step=fp+1)
        
        if best_metrics == None:
            best_metrics = metrics
            best_epoch = fp + 1
        else:
            for metric, metric_value in best_metrics:
                if (str(metric) == label):
                    if (cur_label_value > metric_value):
                        best_metrics = metrics
                        best_epoch = fp + 1
                        break
                    else:
                        break

print(f'Best Epoch: {best_epoch}')
print(best_metrics)
