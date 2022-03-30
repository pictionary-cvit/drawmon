import sys
import os
from xml.etree.ElementTree import tostring
from cv2 import GC_INIT_WITH_MASK
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
from utils.bboxes import iou

import argparse
import json

data_path = "/home/nikhil3456/StdCnetMish/val"
image_path = os.path.join(data_path, "imgs")
metadata_path = os.path.join(data_path, "metadata") # to save jsons
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--data', type=str, required=False, default=data_path)


# load ground-truths and predictions npy files
'''
GTs: n*m*4
n: number of images
m: number of boxes per image(m varies per image)
4: box coords of format (x_min, y_min, x_max, y_max)
'''
GTs = np.load(os.path.join(data_path, "truth.npy"), allow_pickle=True) 
Preds = np.load(os.path.join(data_path, "preds.npy"), allow_pickle=True)

n = GTs.shape[0]
dict = {}
for i in range(n):
    id = f"{i}"
    dict_i = {}
    dict_i[id] = {}
    dict_i[id]["imagePath"] = os.path.join(image_path, f"{id}.png")
    dict_i[id]["regions"] = []

    for j in range(len(GTs[i])):
        dict_j = {}
        bbox = GTs[i][j][0:4]
        bbox = [c*512 for c in bbox]
        x_min, y_min, x_max, y_max = bbox 
        dict_j["groundTruth"] = [[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]

        # TODO: add dict_j["modelPrediction"] for each gt

        dict_j["regionLabel"] = "text"
        dict_j["id"] = f"{id}" 

        dict_i[id]["regions"].append(dict_j)

    dict[id] = dict_i[id]

with open(os.path.join(metadata_path, f"{id}.json"), 'w') as fp:
    json.dump(dict, fp)