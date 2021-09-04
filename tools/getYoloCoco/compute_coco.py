import numpy as np
import os
from tqdm import tqdm

import tensorflow as tf

import argparse
import sys
import glob

from cocoevals import PycocoMetric

import matplotlib.pyplot as plt
import cv2

import json

pycoco_metric = PycocoMetric()

with open('./coco_results.json', 'r') as fd:
    preds = json.load(fd)


category_ids = [1, 2, 3, 4]
confidence_threshold = 0.4

# preds[0]

# from matplotlib.patches import Polygon    
# from PIL import Image

# image_id = 8
# fig, ax = plt.subplots()

# im = np.array(Image.open(f"./obj/{image_id}_val.jpg"))

# ax.imshow(im, cmap='gray')
# ax.axis('off')

# for i in range(len(preds)):
#     if preds[i]['image_id'] == image_id:
#         # print(i)
#         box = preds[i]['bbox']

#         cx, cy, w, h = box

#         y = np.array([[cx - w/2, cy-h/2], [cx+ w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2], [cx-w/2, cy-h/2]])

#         p = Polygon(y, linewidth=1, edgecolor='r', facecolor='none')

#         ax.add_patch(p)
# plt.show()

gt = []

import glob
path = "./obj/*_val.txt"
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        image_id = int(filename.split("/")[-1].split("_")[0])
        # print(image_id)
        for line in f:
            line = line.rstrip("\n")
            vals = line.split(" ")
            category_id = int(vals[0])
            bbox = [float(vals[1])*512.0, float(vals[2])*512.0, float(vals[3])*512.0, float(vals[4])*512.0]
            gt.append({'image_id': image_id, 'category_id': category_id+1, 'bbox': bbox})

dataset = {
    'images': [],
    'annotations': [],
    'categories': []
}

image_ids = []
for i in range(len(gt)):
    image_id = gt[i]['image_id']
    
    if image_id not in image_ids:
        image_ids.append(image_id)
        dataset['images'].append({'id': int(image_id)})
    
    dataset['annotations'].append({
                              'id': i+1,
                              'image_id': image_id,
                              'category_id': gt[i]['category_id'],
                              'bbox': gt[i]['bbox'],
                              'area': gt[i]['bbox'][2]*gt[i]['bbox'][3],
                              'iscrowd': 0
                          })

    
dataset['categories'] = [
    {'id': int(category_id)} for category_id in category_ids
]

detections = []

for i in range(len(preds)):
    if preds[i]['score'] >= confidence_threshold:
        box = preds[i]["bbox"]
        cx, cy, w, h = box[0] + box[2]/2, box[1] + box[3]/2, box[2], box[3]
        final_box = [preds[i]['image_id'], cx, cy, w, h, preds[i]['score'], preds[i]['category_id']] # format imageID, x1, y1, w, h, score, class
        detections.append(final_box)
detections = np.array(detections)

pycoco_metric(dataset, detections)

