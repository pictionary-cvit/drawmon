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
    
    def __init__(self, dataset, prior_util, batch_size, batch_processes, split='train', img_width=512, img_height=512, encode=True, overlap_threshold=0.5, num_classes=5):
        
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



# data_path = "/home/nikhil.bansal/pictionary_redux/pictionary_redux/dataset/obj_detection_data"
# gen = Generator(data_path)
# ds = gen.getDS("train")
# loader = InputGenerator(ds, 8, 5)
# ds = loader.get_dataset()
# for (image, labels) in ds:
#     print (image, labels)
#     break
