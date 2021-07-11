#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import os, time, pickle, sys
from tqdm.notebook import tqdm

from tbpp_model import TBPP512, TBPP512_dense, TBPP512_dense_separable
from tbpp_utils import PriorUtil
from tbpp_training import TBPPFocalLoss

from pictText_utils import Generator
from data_pictText import InputGenerator

# from utils.model import load_weights
from utils.training import MetricUtility

from tbpp_utils import PriorUtil
import argparse

# In[2]:

checkpoint_dir = './checkpoints'
data_path = "/home/nikhil.bansal/pictionary_redux/pictionary_redux/dataset/obj_detection_data"


#### Model

batch_size = 6
experiment = 'tbppDsfl_ct4_sw2_s9_iou5_diouAabb_NoRbb_'
confidence_threshold = 0.3
scale = 0.9
isQuads = 'False'
isRbb='False'

aabb_diou='True'
rbb_diou='False'
num_dense_segs=3 # default = 3
use_prev_feature_map='False' # default = False
num_multi_scale_maps=5 # default = 5
num_classes=1 + 1 # default = 2

lambda_conf=10000.0
lambda_offsets=1.0
aabb_weight=1.0
rbb_weight=1.0
decay_factor = 0.0 # No Decay

isfl='True'
neg_pos_ratio = 3.0

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data', type=str, required=False, default=data_path)
parser.add_argument('--ckpt', type=str, required=False, default=checkpoint_dir)
parser.add_argument('--bs', type=int, required=False, default=batch_size)
parser.add_argument('--exp', type=str, required=False, default=experiment)
parser.add_argument('--ct', type=float, required=False, default=confidence_threshold)
parser.add_argument('--scale', type=float, required=False, default=scale)
parser.add_argument('--isQ', type=eval, 
                      choices=[True, False], required=False, default=isQuads)
parser.add_argument('--isR', type=eval, 
                      choices=[True, False], required=False, default=isRbb)
parser.add_argument('--aDiou', type=eval, 
                      choices=[True, False], required=False, default=aabb_diou)
parser.add_argument('--rDiou', type=eval, 
                      choices=[True, False], required=False, default=rbb_diou)
parser.add_argument('--nds', type=int, required=False, default=num_dense_segs)
parser.add_argument('--isPMap', type=eval, 
                      choices=[True, False], required=False, default=use_prev_feature_map)
parser.add_argument('--nmsm', type=int, required=False, default=num_multi_scale_maps)
parser.add_argument('--nc', type=int, required=False, default=num_classes)
parser.add_argument('--calpha', type=float, required=False, default=lambda_conf)
parser.add_argument('--oalpha', type=float, required=False, default=lambda_offsets)
parser.add_argument('--aWt', type=float, required=False, default=aabb_weight)
parser.add_argument('--rWt', type=float, required=False, default=rbb_weight)
parser.add_argument('--df', type=float, required=False, default=decay_factor)

parser.add_argument('--npr', type=float, required=False, default=neg_pos_ratio)
parser.add_argument('--isfl', type=eval, choices=[True, False], required=False, default=isfl)




args = parser.parse_args()
print(args)

data_path = args.data
checkpoint_dir = args.ckpt

batch_size = args.bs
experiment = args.exp
confidence_threshold = args.ct
scale = args.scale
isQuads = args.isQ
isRbb=args.isR

aabb_diou=args.aDiou
rbb_diou=args.rDiou
num_dense_segs=args.nds # default = 3
use_prev_feature_map=args.isPMap # default = True
num_multi_scale_maps=args.nmsm # default = 5
num_classes=args.nc # default = 1 + 1 (bg + text)

lambda_conf=args.calpha
lambda_offsets=args.oalpha
aabb_weight=args.aWt
rbb_weight=args.rWt
decay_factor = args.df # No Decay

isfl=args.isfl
neg_pos_ratio = args.npr


tf.config.experimental.list_physical_devices()
is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
is_gpu




# TextBoxes++
model = TBPP512_dense_separable(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_dense_segs=num_dense_segs, use_prev_feature_map=use_prev_feature_map, num_multi_scale_maps=num_multi_scale_maps, num_classes=num_classes)
model.num_classes = num_classes
freeze = []
'''
Load weights from previous SSD Model/TBPP Model and you can freeze some layers by defining:
freeze = ['conv1_1', 'conv1_2',
          'conv2_1', 'conv2_2',
          'conv3_1', 'conv3_2', 'conv3_3'
         ]
'''



# In[5]:



prior_util = PriorUtil(model)

checkdir = f'{checkpoint_dir}/batch' + time.strftime('%Y%m%d%H%M') + '_' + experiment

# tb_callback = tf.keras.callbacks.TensorBoard(f"{checkdir}")
# tb_callback.set_model(model)


train_summary_writer = tf.summary.create_file_writer(f"{checkdir}/train")
val_summary_writer = tf.summary.create_file_writer(f"{checkdir}/val")


# In[6]:


# temp
# model.load_weights('./checkpoints/202102161718_tbpp512fl_pictText/weights.007.h5')


# In[12]:


def renderPreds(imgs, preds, truths=None):   
    rends = []
    for i in range(preds.shape[0]):
        res = prior_util.decode(preds[i], confidence_threshold, fast_nms=False)
        res_truth = prior_util.decode(truths[i], confidence_threshold, fast_nms=False)

        fig = plt.figure(figsize=[8]*2)
        im = np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1]))
        plt.imshow(im, cmap='gray')
        # prior_util.plot_gt()
        prior_util.plot_results(res, gt_data_decoded=res_truth, show_labels=True)
        plt.axis('off')
        # plt.show()
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        rends.append(data)

        plt.close('all')
    
    return np.array(rends)


# ### Training

# In[ ]:


epochs = 100
initial_epoch = 0

#optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)


priors_xy = tf.Variable(prior_util.priors_xy/prior_util.image_size, dtype=tf.float32)
priors_wh = tf.Variable(prior_util.priors_wh/prior_util.image_size, dtype=tf.float32)
priors_variances = tf.Variable(prior_util.priors_variances, dtype=tf.float32)


loss = TBPPFocalLoss(lambda_conf=lambda_conf, lambda_offsets=lambda_offsets, isQuads=isQuads, isRbb=isRbb, aabb_weight=aabb_weight, rbb_weight=rbb_weight, decay_factor = decay_factor, priors_xy=priors_xy, priors_wh=priors_wh, priors_variances=priors_variances, aabb_diou=aabb_diou, rbb_diou=rbb_diou, isfl=isfl, neg_pos_ratio=neg_pos_ratio)

#regularizer = None
regularizer = keras.regularizers.l2(5e-4) # None if disabled

gen = Generator(data_path, padding=0)
ds_train = gen.getDS("train", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10)
ds_val = gen.getDS("val", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=10)

gen_train = InputGenerator(ds_train, prior_util, batch_size, 5, num_classes=num_classes)
gen_val = InputGenerator(ds_val, prior_util, batch_size, 5, num_classes=num_classes)

iterator_train = gen_train.get_dataset()
iterator_val = gen_val.get_dataset()


if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(f"{checkdir}/modelsummary.txt", 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

tf.keras.utils.plot_model(
    model, to_file=f"{checkdir}/model.png", show_shapes=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
)

with open(f"{checkdir}/modelargs.txt", "w") as f:
    print(args, file=f)

'''
with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
'''

print(checkdir)

x = []
y_true = []
for l in model.layers:
    l.trainable = not l.name in freeze
    if regularizer and l.__class__.__name__.startswith('Conv'):
        model.add_loss(lambda l=l: regularizer(l.kernel))

metric_util = MetricUtility(loss.metric_names, logdir=checkdir)

iteration = 0

# @tf.function
def step(x, y_true, training=False):
    if training:
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            metric_values = loss.compute(y_true, y_pred)
            total_loss = metric_values['loss']
            if len(model.losses):
                total_loss += tf.add_n(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        y_pred = model(x, training=True)
        metric_values = loss.compute(y_true, y_pred)
    return metric_values

for k in tqdm(range(initial_epoch, epochs), 'total', leave=False):
    print('\nepoch %i/%i' % (k+1, epochs))
    metric_util.on_epoch_begin()

    for i, train_sample in enumerate(iterator_train):
        try:
            x, y_true = train_sample
        except:
            print("Skipping Sample: ", i)
            continue
        print(x.shape)
        metric_values = step(x, y_true, training=True)
        metric_util.update(metric_values, training=True)
        
        with train_summary_writer.as_default():
            for metric in metric_values:
                tf.summary.scalar(str(metric), metric_values[metric], step=iteration)
        iteration += 1
        
    
    model.save_weights(checkdir+'/weights.%03i.h5' % (k+1,))

    val_loss = None
    num_val_batches = 0
    
    min_val_loss = None
    best_batch_images = None
    best_batch_gt = None
    best_batch_preds = None
    
    max_val_loss = None
    worst_batch_images = None
    worst_batch_gt = None
    worst_batch_preds = None
    
    first_val_loss = None
    first_batch_images = None                               
    first_batch_gt = None                                   
    first_batch_preds = None
    
    for i, val_sample in enumerate(iterator_val):
        try:
            x, y_true = val_sample
        except:
            print("Skipping val Sample: ", i)
            continue

        metric_values = step(x, y_true, training=False)
        if not val_loss:
            val_loss = metric_values
        else:
            for key in val_loss:
                val_loss[key] += metric_values[key]
        
        if not min_val_loss:
            min_val_loss = metric_values['loss']
            best_batch_images = x
            best_batch_gt = y_true
            best_batch_preds = model.predict(x, batch_size=batch_size, verbose=1)
        else:
            if min_val_loss > metric_values['loss']:
                min_val_loss = metric_values['loss']
                best_batch_images = x
                best_batch_gt = y_true
                best_batch_preds = model.predict(x, batch_size=batch_size, verbose=1)
        
        if not first_val_loss:                                
            first_val_loss = metric_values['loss']            
            first_batch_images = x                          
            first_batch_gt = y_true                         
            first_batch_preds = model.predict(x, batch_size=batch_size, verbose=1)


        if not max_val_loss:
            max_val_loss = metric_values['loss']
            worst_batch_images = x
            worst_batch_gt = y_true
            worst_batch_preds = model.predict(x, batch_size=batch_size, verbose=1)
        else:
            if max_val_loss < metric_values['loss']:
                max_val_loss = metric_values['loss']
                worst_batch_images = x
                worst_batch_gt = y_true
                worst_batch_preds = model.predict(x, batch_size=batch_size, verbose=1)
                
        
        num_val_batches += 1
        metric_util.update(metric_values, training=False)
        
    
    with val_summary_writer.as_default():
        for metric in val_loss:
            tf.summary.scalar(str(metric), val_loss[metric]/num_val_batches, step=iteration)
        
        first_renders = renderPreds(first_batch_images, first_batch_preds, first_batch_gt)
        best_renders = renderPreds(best_batch_images, best_batch_preds, best_batch_gt)
        worst_renders = renderPreds(worst_batch_images, worst_batch_preds, worst_batch_gt)
        
        tf.summary.image(f"first_val_images", first_renders, step=iteration)
        tf.summary.image(f"best_val_images", best_renders, step=iteration)
        tf.summary.image(f"worst_val_images", worst_renders, step=iteration)

    metric_util.on_epoch_end(verbose=1)


# In[ ]:
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')
    

model.save('./saved_models/my_model' + '_' + time.strftime('%Y%m%d%H%M') + '_' + experiment)


# In[16]:






# In[ ]:




