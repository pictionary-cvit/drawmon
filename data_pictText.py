from functools import reduce
import itertools
from random import random
from re import S
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import tensorflow as tf
import math

from tensorflow._api.v2 import data

# from pictText_utils import Generator

def convertToCentroid(bb):
    """
        bb: (4 coords = 8 values) 4 x 2
        return cx, cy, w, h
    """
    bb = np.array(bb)

    mx, my = np.min(bb, axis=0)
    Mx, My = np.max(bb, axis=0)

    cx = (mx+Mx)/2
    cy = (my+My)/2

    w = Mx - mx
    h = My - my

    return np.array((cx, cy, w, h))


class InputGenerator(object):
    """Model input generator for data augmentation."""

    # TODO
    # flag to protect bounding boxes from cropping?
    # crop range > 1.0? crop_area_range=[0.75, 1.25]

    def __init__(
        self,
        dataset,
        prior_util,
        batch_size,
        batch_processes,
        split="train",
        img_width=512,
        img_height=512,
        encode=True,
        overlap_threshold=0.5,
        num_classes=2,
        isFlattenBoxes=True, # if true, change shape of each box from Nx4x2 to Nx8
        isConvertToCentroid=False
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_processes = batch_processes
        self.img_width = img_width
        self.img_height = img_height
        self.prior_util = prior_util
        self.split = split
        self.encode = encode
        self.overlap_threshold = overlap_threshold
        self.num_classes = num_classes

        # if true, change shape of each box from Nx4x2 to Nx8
        self.isFlattenBoxes=isFlattenBoxes

         # if true, convert format to cx, cy, w, h (eg. true in case of darknet format)
        self.isConvertToCentroid = isConvertToCentroid

    def stackBatch(self, batch):
        if self.num_classes == 2:
            imgs, boxes = list(zip(*batch))
        else:
            # multiclass
            imgs, box_datas = list(zip(*batch))
            boxes, anomaly_classes = list(zip(*box_datas))

        imgs = [
            np.reshape(img.astype(np.float32), (img.shape[0], img.shape[1], 1))
            for img in imgs
        ]

        targets = []
        for idx, target in enumerate(boxes):
            target = np.array(target, dtype="float32")
            target[:, :, 0] = target[:, :, 0] / self.img_width
            target[:, :, 1] = target[:, :, 1] / self.img_height
            if (self.isFlattenBoxes): target = target.reshape(target.shape[0], -1)
            if (self.isConvertToCentroid and self.isFlattenBoxes == False):
                new_target = []
                for i in range(target.shape[0]):
                    new_target.append(convertToCentroid(target[i]))
                new_target = np.array(new_target) #n x 4
                target = new_target

            # append class 1 => text class
            if self.num_classes == 2:
                target = np.concatenate([target, np.ones([target.shape[0], 1])], axis=1)
            else:
                # multiclass
                target = np.concatenate(
                    [target, np.array(anomaly_classes[idx])[:, None]], axis=1
                )

            if self.encode:
                target = self.prior_util.encode(
                    target,
                    overlap_threshold=self.overlap_threshold,
                    num_classes=self.num_classes,
                )

            targets.append(target)

        return np.array(imgs, dtype="float32"), np.array(targets, dtype="float32")

    def get_dataset(self):
        if self.split == "train":
            ds = (
                self.dataset.shuffle()
                .batch(self.batch_size, self.batch_processes)
                .map(self.stackBatch)
            )
        else:
            ds = self.dataset.batch(self.batch_size, self.batch_processes).map(
                self.stackBatch
            )

        return ds


class ImageInputGenerator(object):
    """Model input generator with images i.e without using memcache"""

    def __init__(self, data_path, batch_size, dataset="train", give_idx=False, anomaly_class=None, create_batch=True):
        
        if anomaly_class is not None:
            self.data_path = os.path.join(data_path, dataset, anomaly_class)
            assert(give_idx == False)
        else: self.data_path = os.path.join(data_path, dataset)
        
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_samples = len(list(set(glob.glob1(self.data_path, "*.npy")))) // 2
        self.give_idx = give_idx
        self.create_batch = create_batch

        self.samples = [i for i in range(self.num_samples)]
        random.shuffle(self.samples)

    def get_sample(self, idx):
        idx = self.samples[idx]
        img = np.load(os.path.join(self.data_path, f"sample_{idx}.npy"))
        y = np.load(os.path.join(self.data_path, f"label_{idx}.npy"))

        if self.give_idx:
            return img, y, int(idx)
        else:
            return img, y

    def get_dataset(self, num_parallel_calls=1, seed=1337):
        import tensorflow as tf

        print(
            f"Number of {self.dataset} samples at '{self.data_path}': {self.num_samples}"
        )

        if seed is not None:
            np.random.seed(seed)

        print(f"Number of samples: {len(self.samples)}")

        assert len(self.samples) >= self.batch_size

        mod = (len(self.samples)) % (self.batch_size)
        self.samples = self.samples + self.samples[: (self.batch_size - mod)]

        assert (len(self.samples)) % (self.batch_size) == 0


        type = None
        if self.give_idx:
            type = ["float32", "float32", "int64"]
        else:
            type = ["float32", "float32"]

        ds = tf.data.Dataset.range(len(self.samples)).repeat(1).shuffle(len(self.samples))
        ds = ds.map(
            lambda x: tf.py_function(
                self.get_sample,
                [
                    x,
                ],
                type,
            ),
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )

        if self.create_batch:
            ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	
        print(len(ds))

        return ds


class ImageInputGeneratorMulticlass(object):
    """Model input generator with images i.e without using memcache"""

    def __init__(self, data_path, batch_size, split, give_idx=False):
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.num_samples = len(list(set(glob.glob(os.path.join(self.data_path, self.split, "**", "*.png"), recursive=True))))
        self.give_idx = give_idx

    @staticmethod
    def getFileList(
        dataset,
        split,
        anomaly_class="**",
    ):
        print(f"getting list of (sample, label) in {os.path.join(dataset, split, anomaly_class)}........")
        pattern = os.path.join(dataset, split, anomaly_class, "label_*.npy")
        labels = glob.glob(pattern, recursive=True)

        def getImageName(label_file):
            dir_name = "/".join(label_file.split("/")[:-1])
            number = label_file.split("/")[-1].split(".")[0].split("_")[-1]
            filename = f"sample_{number}.npy"
            return os.path.join(dir_name, filename), label_file

        labels = list(set(labels))
        return list(map(getImageName, labels))

    def createDSFromFiles(self, files, repeat):
        print("creating DS from files........")
        if self.give_idx:
            type = ["float32", "float32", "string", "string"]
        else:
            type = ["float32", "float32"]
        
        ds = tf.data.Dataset.from_tensor_slices(files).repeat(repeat).map(
            lambda x: tf.py_function(self.get_sample, [x[0], x[1]], type),
            num_parallel_calls=1,
            deterministic=False,
        )
        print(len(ds))
        # print(list(ds.as_numpy_iterator()))

        return (
            ds 
        )

    def get_sample(self, image_file, label_file):
        # print(image_file, label_file)
        img = np.load(image_file.numpy())
        y = np.load(label_file.numpy())

        if self.give_idx:
            return img, y, image_file, label_file
        else:
            return img, y

    def get_dataset(self, seed=1337):
        import tensorflow as tf

        circle = ImageInputGeneratorMulticlass.getFileList(
            self.data_path, split=self.split, anomaly_class="circle"
        )
        number = ImageInputGeneratorMulticlass.getFileList(
            self.data_path, split=self.split, anomaly_class="number"
        )
        text = ImageInputGeneratorMulticlass.getFileList(
            self.data_path, split=self.split, anomaly_class="text"
        )
        symbol = ImageInputGeneratorMulticlass.getFileList(
            self.data_path, split=self.split, anomaly_class="symbol"
        )
        real = ImageInputGeneratorMulticlass.getFileList(
            self.data_path, split=self.split, anomaly_class="real"
        )

        files = list(
            map(
                lambda x: x[: len(x) // self.batch_size * self.batch_size],
                [circle, number, text, symbol, real],
            )
        )
        lens = list(map(lambda x: len(x), files))
        repeats = list(map(lambda x: max(lens) // x, lens))

        print(
            f"""Number of {self.split} samples at '{self.data_path}': {self.num_samples}
            Number of samples collected = {list(map(len, files))}
            Dataset = [circle, number, text, symbol, real]
            repeats = {repeats}
            """
        )

        if seed is not None:
            np.random.seed(seed)

        # option-1 #####################
        datasets = list(map(lambda x: self.createDSFromFiles(*x), zip(files, repeats)))
        final_dataset = tf.data.experimental.sample_from_datasets(datasets)
        ################################

        # # option-2 #####################
        # all_files = []
        # for i, fp in enumerate(files): all_files += fp*repeats[i]   
        # if self.give_idx:
        #     type = ["float32", "float32", "string", "string"]
        # else:
        #     type = ["float32", "float32"]

        # final_dataset = tf.data.Dataset.from_tensor_slices(all_files).repeat(1).shuffle(len(all_files)).map(
        #     lambda x: tf.py_function(self.get_sample, [x[0], x[1]], type),
        #     num_parallel_calls=1,
        #     deterministic=False,
        # ) 
        # print(len(final_dataset))
        # #################################
      
        return final_dataset.batch(self.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
        )


class ImageInputGeneratorWithResampling(object):
    """
    Model input generator with more weights to hard examples
    """

    def __init__(self, data_path, batch_size, dataset="train"):
        self.data_path = os.path.join(data_path, dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_samples = len(list(set(glob.glob1(self.data_path, "*.png"))))

    def get_sample(self, idx):
        img = np.load(os.path.join(self.data_path, f"sample_{idx}.npy"))
        y = np.load(os.path.join(self.data_path, f"label_{idx}.npy"))

        return img, y, int(idx)

    def get_dataset(
        self,
        num_parallel_calls=1,
        seed=1337,
        hard_examples=[],
        normal_examples=[],
        max_repeat=3,
    ):
        import tensorflow as tf

        if len(hard_examples) == 0:
            normal2hard_ratio = 1
        else:
            normal2hard_ratio = max(len(normal_examples) // len(hard_examples), 1)

        print(
            f"Number of {self.dataset} samples at '{self.data_path}': {self.num_samples}"
        )

        if seed is not None:
            np.random.seed(seed)

        ds_type = ["float32", "float32", "int64"]

        print(f"Number of hard examples: {len(hard_examples)}")
        print(f"Number of normal examples: {len(normal_examples)}")

        samples = min(max_repeat, normal2hard_ratio) * hard_examples + normal_examples

        print("assert if hard_examples + normal_examples == num_samples")
        assert sum(range(self.num_samples)) == (
            sum(hard_examples) + sum(normal_examples)
        )

        print(f"Number of samples: {len(samples)}")

        assert len(samples) >= self.batch_size

        mod = (len(samples)) % (self.batch_size)
        samples = samples + samples[: (self.batch_size - mod)]

        assert (len(samples)) % (self.batch_size) == 0

        ds = tf.data.Dataset.from_tensor_slices(samples).repeat(1).shuffle(len(samples))
        ds = ds.map(
            lambda x: tf.py_function(
                self.get_sample,
                [
                    x,
                ],
                ds_type,
            ),
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
        ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds


class ImageInputGeneratorForCurriculumTraining(object):
    """Model input generator with images i.e without using memcache"""

    def __init__(
        self, 
        data_path, 
        batch_size,
        img_wd,
        img_ht,
        lower_area_thres,
        upper_area_thres,
        prior_util, 
        ct,
        class_idx=-1,
        fast_nms=False,
        dataset="train", 
        give_idx=False, 
        anomaly_class=None, 
        create_batch=True):
        
        if anomaly_class is not None:
            self.data_path = os.path.join(data_path, dataset, anomaly_class)
            assert(give_idx == False)
        else: self.data_path = os.path.join(data_path, dataset)
        
        self.IDs = []

        self.batch_size = batch_size
        self.lower_area_thres = lower_area_thres
        self.upper_area_thres = upper_area_thres
        self.img_wd = img_wd
        self.img_ht = img_ht
        self.ct = ct
        self.class_idx = class_idx
        self.fast_nms = fast_nms
        self.prior_util = prior_util
        self.dataset = dataset
        self.num_samples = len(list(set(glob.glob1(self.data_path, "*.npy")))) // 2
        self.give_idx = give_idx
        self.create_batch = create_batch

    def get_sample(self, idx):
        idx = self.IDs[idx]
        img = np.load(os.path.join(self.data_path, f"sample_{idx}.npy"))
        y = np.load(os.path.join(self.data_path, f"label_{idx}.npy"))

        return img, y, int(idx)
    
    def get_curr_area_samples(self):
        
        for idx in range(self.num_samples):
            img = np.load(os.path.join(self.data_path, f"sample_{idx}.npy"))
            y = np.load(os.path.join(self.data_path, f"label_{idx}.npy"))

            decoded_y = self.prior_util.decode(
                y, 
                class_idx=self.class_idx, 
                confidence_threshold=self.ct, 
                fast_nms=self.fast_nms)
            
            is_curr_area = False
            for i in range(len(decoded_y)):
                box = decoded_y[i][0:4]*(self.img_wd, self.img_ht, self.img_wd, self.img_ht)
                area = (box[2]-box[0])*(box[3]-box[1])
                if area >= self.lower_area_thres and area <= self.upper_area_thres:
                    is_curr_area = True

            if is_curr_area:
                self.IDs.append(idx)
        
    def get_dataset(self, num_parallel_calls=1, seed=1337):
        import tensorflow as tf

        print(
            f"Number of {self.dataset} samples at '{self.data_path}': {self.num_samples}"
        )

        if seed is not None:
            np.random.seed(seed)

        type = None
        if self.give_idx:
            type = ["float32", "float32", "int64"]
        else:
            type = ["float32", "float32"]

        self.get_curr_area_samples()

        ds = tf.data.Dataset.range(len(self.IDs)).repeat(1).shuffle(len(self.IDs))
        ds = ds.map(
            lambda x: tf.py_function(
                self.get_sample,
                [
                    x,
                ],
                type,
            ),
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )

        if self.create_batch:
            ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	
        print(len(ds))

        return ds
