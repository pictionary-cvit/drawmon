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
from data_pictText import ImageInputGenerator, ImageInputGeneratorWithResampling

# from utils.model import load_weights
from utils.training import MetricUtility
from utils.bboxes import iou

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
activation = 'relu'

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
parser.add_argument('--activation', type=str, required=False, default='relu')
parser.add_argument('--wlb', type=float, required=False, default=0.45)
parser.add_argument('--wub', type=float, required=False, default=0.55)



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
activation = args.activation

# window size for hard example classification
window_size_lb = args.wlb
window_size_ub = args.wub

tf.config.experimental.list_physical_devices()
is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
is_gpu

fl_alpha = [0.002, 0.998]
if num_classes == 5:
    fl_alpha=[0.002, 0.11721553, 0.28019262, 0.27949907, 0.32109278]

mirrored_strategy = tf.distribute.MirroredStrategy()

# TextBoxes++
with mirrored_strategy.scope():
    model = TBPP512_dense_separable(input_shape=(512, 512, 1), softmax=True, scale=scale, isQuads=isQuads, isRbb=isRbb, num_dense_segs=num_dense_segs, use_prev_feature_map=use_prev_feature_map, num_multi_scale_maps=num_multi_scale_maps, num_classes=num_classes, activation=activation)

    # optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
    optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)


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


priors_xy = tf.Variable(prior_util.priors_xy/prior_util.image_size, dtype=tf.float32)
priors_wh = tf.Variable(prior_util.priors_wh/prior_util.image_size, dtype=tf.float32)
priors_variances = tf.Variable(prior_util.priors_variances, dtype=tf.float32)


loss = TBPPFocalLoss(lambda_conf=lambda_conf, lambda_offsets=lambda_offsets, isQuads=isQuads, isRbb=isRbb, aabb_weight=aabb_weight, rbb_weight=rbb_weight, decay_factor = decay_factor, priors_xy=priors_xy, priors_wh=priors_wh, priors_variances=priors_variances, aabb_diou=aabb_diou, rbb_diou=rbb_diou, isfl=isfl, neg_pos_ratio=neg_pos_ratio, alpha=fl_alpha)

# regularizer = None
regularizer = keras.regularizers.l2(5e-4) # None if disabled

hard_examples = []
normal_examples = []

def is_hard_example(gt, pred):
    """
    if any box in gt, lets say b1 has maximum overlap threshold within window around 0.5 with any of the correctly classified prediction boxes
    # Arguments
        gt, pred: bounding boxes, numpy array of 
            shape (num_boxes, 4).
        Bounding box should be a numpy array of shape (4).
            (x1, y1, x2, y2)
    """
    for b1 in gt:
        max_overlap = 0.0
        ious = iou(b1, pred)
        for val in ious:
            if (val > max_overlap): max_overlap = val
        if window_size_lb <= max_overlap <= window_size_ub:
            return True
        
    return False
    

    
def divide_train_dataset(gts, preds, idxs):
    """
    Mine hard examples after each epoch
    - gts: array of ground truth samples of size (BX...)
    - preds: predictions array (BX...)
    - idxs: indexes of gts of size (B)
    """
    step_hard_examples = []
    step_normal_examples = []
    
    for i, gt in enumerate(gts):
        # classify sample
        decoded_gt = prior_util.decode(gt, class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False) # class_idx = -1 => all classes
        decoded_pred = prior_util.decode(preds[i], class_idx = -1, confidence_threshold = confidence_threshold, fast_nms=False)
        
        idx = idxs[i]
        if (is_hard_example(decoded_gt[:,:4], decoded_pred[:,:4])): step_hard_examples.append(idx)
        else: step_normal_examples.append(idx)
    
    return step_hard_examples, step_normal_examples

def make_train_dataset():
    """
    Make train dataset with hard samples mining
    """
    gen_train = ImageInputGeneratorWithResampling(data_path, batch_size, 'train', hard_examples, normal_examples)
    dataset_train = gen_train.get_dataset()
    dist_dataset_train = mirrored_strategy.experimental_distribute_dataset(dataset_train)
    
    return dist_dataset_train
        
    

gen_train = ImageInputGenerator(data_path, batch_size, 'train', give_idx=True)
gen_val = ImageInputGenerator(data_path, batch_size, 'val', give_idx=True)

dataset_train, dataset_val = gen_train.get_dataset(), gen_val.get_dataset()

dist_dataset_train = mirrored_strategy.experimental_distribute_dataset(dataset_train)
dist_dataset_val = mirrored_strategy.experimental_distribute_dataset(dataset_val)

# iterator_train = iter(dataset_train)
# iterator_val = iter(dataset_val)


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

iteration = 0

# @tf.function
def step(inputs, training=True):
    x, y_true, idx = inputs
    if training:
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            print(f"Input shape: {x.shape}")
            print(f"GT Shape: {y_true.shape}")
            print(f"Pred Shape: {y_pred.shape}")
            metric_values = loss.compute(y_true, y_pred)
            total_loss = metric_values['loss']
            if len(model.losses):
                total_loss += tf.add_n(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        replica_hard_examples, replica_normal_examples = divide_train_dataset(y_true, y_pred, idx)
        
        return total_loss, replica_hard_examples, replica_normal_examples
        
    else:
        y_pred = model(x, training=True)
        metric_values = loss.compute(y_true, y_pred)
        total_loss = metric_values['loss']
        if len(model.losses):
            total_loss += tf.add_n(model.losses)
        return total_loss

# @tf.function
def distributed_train_step(dist_inputs):
    per_replica_results = mirrored_strategy.run(step, args=(dist_inputs, True,))
    list_results = mirrored_strategy.experimental_local_results(per_replica_results)
    
    total_loss = 0.0
    
    for (replica_loss, replica_hard_examples, replica_normal_examples) in list_results:
        total_loss += replica_loss
        hard_examples += replica_hard_examples
        normal_examples += replica_normal_examples
        
    return total_loss
    
    

# @tf.function
def distributed_val_step(dist_inputs):
    per_replica_losses = mirrored_strategy.run(step, args=(dist_inputs, False,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for k in tqdm(range(initial_epoch, epochs), 'total', leave=False):
    print('\nepoch %i/%i' % (k+1, epochs))
    

    for dist_inputs in dist_dataset_train:
        batch_loss = distributed_train_step(dist_inputs)
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', batch_loss, step=iteration)
        iteration += 1
    
    dist_dataset_train = make_train_dataset()
    
    hard_examples = []
    normal_examples = []
    
    
    model.save_weights(checkdir+'/weights.%03i.h5' % (k+1,))
   
    num_val_batches = 0
    val_loss = 0.0
    for dist_inputs in dist_dataset_val:
        batch_loss = distributed_val_step(dist_inputs)
        val_loss += batch_loss
        num_val_batches += 1
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss/num_val_batches, step=iteration) 



# In[ ]:
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')
    

model.save('./saved_models/my_model' + '_' + time.strftime('%Y%m%d%H%M') + '_' + experiment)


# In[16]:






# In[ ]:


