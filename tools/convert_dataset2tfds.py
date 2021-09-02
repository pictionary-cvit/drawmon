import sys
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import tensorflow as tf

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable
from tbpp_utils import PriorUtil

from pictText_utils import Generator
from data_pictText import InputGenerator

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

data_path = "/home/sparsh.garg/nikhil/pictionary_redux/dataset/data_multiclass_circle"
batch_size = 8
save_path = "../saved_datasets"
scale = 0.9
isQuads = False
isRbb = False
num_classes = 2 # includes background class

num_dense_segs=3 # default = 3
use_prev_feature_map='False' # default = False
num_multi_scale_maps=5 # default = 5

parser = argparse.ArgumentParser(description='Hyperparameters')

parser.add_argument('--data', type=str, required=False, default=data_path)
parser.add_argument('--nc', type=int, required=False, default=num_classes)
parser.add_argument('--bs', type=int, required=False, default=batch_size)
parser.add_argument('--ct', type=float, required=False, default=0.4)
parser.add_argument('--tsave', type=str, required=False, default=save_path)
parser.add_argument('--vsave', type=str, required=False, default=save_path)
parser.add_argument('--scale', type=float, required=False, default=scale)
parser.add_argument('--isQ', type=eval, choices=[True, False], required=False, default=isQuads)
parser.add_argument('--isR', type=eval, choices=[True, False], required=False, default=isRbb)
parser.add_argument('--nds', type=int, required=False, default=num_dense_segs)
parser.add_argument('--isPMap', type=eval, 
                      choices=[True, False], required=False, default=use_prev_feature_map)
parser.add_argument('--nmsm', type=int, required=False, default=num_multi_scale_maps)

# data-type
parser.add_argument('--RorA', type=str, required=False, default="real")
parser.add_argument('--className', type=str, required=False, default="text")
# parser.add_argument('--class', type=str, required=False, default="text")

# whether to make val dataset
parser.add_argument('--mVal', type=eval,
                      choices=[True, False], required=False, default='True')

parser.add_argument('--baseline', type=str, required=False, default="tbppds")

args = parser.parse_args()
print(args)

data_path = args.data
batch_size = args.bs
confidence_threshold = args.ct
train_save_path = args.tsave
val_save_path = args.vsave
scale = args.scale
isQuads = args.isQ
isRbb = args.isR
num_classes = args.nc # includes background class

num_dense_segs=args.nds # default = 3
use_prev_feature_map=args.isPMap # default = False
num_multi_scale_maps=args.nmsm # default = 5

augmented = args.RorA
class_name = args.className

makeVal = args.mVal
baseline = args.baseline

if baseline == "tbppds":
    model = TBPP512_dense_separable(input_shape=(512, 512, 1), scale=scale, isQuads=isQuads, isRbb=isRbb, num_dense_segs=num_dense_segs, use_prev_feature_map=use_prev_feature_map, num_multi_scale_maps=num_multi_scale_maps, num_classes=num_classes)
if baseline == "tbpp":
    model = TBPP512(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes) 
if baseline == "dsod":
    model = TBPP512_dense(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_classes=num_classes)
    

# with open(f"{save_path}/model_args.txt", "w") as f:
#    print(args, file=f)

prior_util = PriorUtil(model)

gen = Generator(data_path, padding=0)
ds_train = gen.getDS("train", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, num_classes=num_classes, augmented=augmented, class_name=class_name)
ds_val = gen.getDS("val", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10, augmented="**", num_classes=num_classes)

gen_train = InputGenerator(ds_train, prior_util, batch_size, 5, num_classes=num_classes)
gen_val = InputGenerator(ds_val, prior_util, batch_size, 5, num_classes=num_classes)

iterator_train = gen_train.get_dataset()
iterator_val = gen_val.get_dataset()

train_examples = None
train_labels = None
count = 0

classes = ["bg", "text", "number", "symbol", "circle"]

def renderPreds(imgs, truths=None):   
    rends = []
    for i in range(imgs.shape[0]):
        res_truth = prior_util.decode(truths[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)

        fig = plt.figure(figsize=[9]*2)
        im = np.pad(np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])), pad_width=(15), constant_values=(0))
        print(im.shape)
        plt.imshow(1-im, cmap='gray')
        
        prior_util.plot_results([], gt_data_decoded=res_truth, show_labels=False, classes=classes, hw = (imgs[i].shape[0], imgs[i].shape[1]), pad=15)

        plt.axis('off')
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')

    return np.array(rends)


if not os.path.exists(f"{train_save_path}"):
    os.makedirs(f"{train_save_path}")

if makeVal:
    if not os.path.exists(f"{val_save_path}"):
        os.makedirs(f"{val_save_path}")


for i, train_sample in enumerate(iterator_train):
    try:
        x, y_true = train_sample
        # train_examples = np.concatenate((train_examples, x), axis=0)
        # train_labels = np.concatenate((train_labels, y_true), axis=0)
        # rends = renderPreds(x, y_true)
    
        for i in range(x.shape[0]):
            np.save(f"{train_save_path}/sample_{count}.npy", x[i])
            tf.keras.preprocessing.image.save_img(f"{train_save_path}/sample_{count}.png", x[i], scale=False)
            # cv2.imwrite(f"{train_save_path}/sample_{count}.png", rends[i])
            # img = tf.keras.preprocessing.image.load_img(f"{train_save_path}/{count}.png", grayscale=True)
            # tf.keras.preprocessing.image.img_to_array(img)
            np.save(f"{train_save_path}/label_{count}.npy", y_true[i])
            count += 1
    except:
        print("Skipping Sample: ", i)
        continue

if makeVal == False:
    pass
else:
    count = 0
    for i, val_sample in enumerate(iterator_val):
        try:
            x, y_true = val_sample

            # rends = renderPreds(x, y_true)

            for i in range(x.shape[0]):
                np.save(f"{val_save_path}/sample_{count}.npy", x[i])
                tf.keras.preprocessing.image.save_img(f"{val_save_path}/sample_{count}.png", x[i], scale=False)
                # cv2.imwrite(f"{val_save_path}/sample_{count}.png", rends[i])
                np.save(f"{val_save_path}/label_{count}.npy", y_true[i])
                count += 1
        except:
            print("Skipping Sample: ", i)
            continue

# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# print(train_dataset.element_spec)
# tf.data.experimental.save(train_dataset, "./saved_datasets/") 
