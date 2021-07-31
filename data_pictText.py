import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import tensorflow as tf

from pictText_utils import Generator



class InputGenerator(object):
    """Model input generator for data augmentation."""
    # TODO
    # flag to protect bounding boxes from cropping?
    # crop range > 1.0? crop_area_range=[0.75, 1.25]
    
    def __init__(self, dataset, prior_util, batch_size, batch_processes, split='train', img_width=512, img_height=512, encode=True, overlap_threshold=0.5, num_classes=2):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_processes = batch_processes
        self.img_width = img_width
        self.img_height = img_height
        self.prior_util = prior_util
        self.split = split
        self.encode = encode
        self.overlap_threshold = overlap_threshold
        self.num_classes=num_classes
    
    def stackBatch(self, batch):
        if self.num_classes == 2:
            imgs, boxes = list(zip(*batch))
        else:
            # multiclass
            imgs, box_datas = list(zip(*batch))
            boxes, anomaly_classes = list(zip(*box_datas))

        imgs = [np.reshape(img.astype(np.float32), (img.shape[0], img.shape[1], 1)) for img in imgs]

        targets = []
        for idx, target in enumerate(boxes):
            target = np.array(target, dtype='float32')
            target[:,:,0] = target[:,:,0]/self.img_width
            target[:,:,1] = target[:,:,1]/self.img_height
            target = target.reshape(target.shape[0], -1)
            
            # append class 1 => text class
            if self.num_classes == 2:
                target = np.concatenate([target, np.ones([target.shape[0],1])], axis=1)
            else:
                # multiclass
                target = np.concatenate([target, np.array(anomaly_classes[idx])[:, None]], axis=1)
            
            if self.encode:
                target = self.prior_util.encode(target, overlap_threshold=self.overlap_threshold, num_classes=self.num_classes)
                
            targets.append(target)

        return np.array(imgs, dtype='float32'), np.array(targets, dtype='float32')    

    
    def get_dataset(self):
        if (self.split == 'train'):
            ds = self.dataset.shuffle().batch(self.batch_size, self.batch_processes).map(self.stackBatch)
        else:
            ds = self.dataset.batch(self.batch_size, self.batch_processes).map(self.stackBatch)
        
        return ds


class ImageInputGenerator(object):
    """Model input generator with images i.e without using memcache"""

    def __init__(self, data_path, batch_size, dataset='train', give_idx=False):
        self.data_path = os.path.join(data_path, dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_samples = len(glob.glob1(self.data_path, "*.png"))
        self.give_idx = give_idx

    
    def get_sample(self, idx):
        img = np.load(os.path.join(self.data_path, f"sample_{idx}.npy"))
        y = np.load(os.path.join(self.data_path, f"label_{idx}.npy"))
        
        if self.give_idx: return img, y, idx
        else: return img, y
    
    def get_dataset(self, num_parallel_calls=1, seed=1337):
        import tensorflow as tf
       
        print(f"Number of {self.dataset} samples at '{self.data_path}': {self.num_samples}")
 
        if seed is not None:
            np.random.seed(seed)
        
        type = None
        if self.give_idx: type = ['float32', 'float32', 'int32']
        else: type = ['float32', 'float32']
        
        ds = tf.data.Dataset.range(self.num_samples).repeat(1).shuffle(self.num_samples)
        ds = ds.map(lambda x: tf.py_function(self.get_sample, [x,], type), num_parallel_calls=num_parallel_calls, deterministic=False)
        ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds
