from functools import reduce
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import tensorflow as tf
import math

from tensorflow._api.v2 import data

# from pictText_utils import Generator


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
            target = target.reshape(target.shape[0], -1)

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

    def __init__(self, data_path, batch_size, dataset="train", give_idx=False):
        self.data_path = os.path.join(data_path, dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_samples = len(glob.glob1(self.data_path, "*.png"))
        self.give_idx = give_idx

    def get_sample(self, idx):
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

        type = None
        if self.give_idx:
            type = ["float32", "float32", "int64"]
        else:
            type = ["float32", "float32"]

        ds = tf.data.Dataset.range(self.num_samples).repeat(1).shuffle(self.num_samples)
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
        ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds


class ImageInputGeneratorMulticlass(object):
    """Model input generator with images i.e without using memcache"""

    def __init__(self, data_path, batch_size, split, give_idx=False):
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.num_samples = len(glob.glob1(self.data_path, "*.png"))
        self.give_idx = give_idx

    @staticmethod
    def getFileList(
        dataset,
        split,
        anomaly_class="**",
    ):
        pattern = os.path.join(dataset, split, anomaly_class, "label_*.npy")
        labels = glob.glob(pattern, recursive=True)

        def getImageName(label_file):
            dir_name = "/".join(label_file.split("/")[:-1])
            number = label_file.split("/")[-1].split(".")[0].split("_")[-1]
            filename = f"sample_{number}.npy"
            return os.path.join(dir_name, filename)

        imgs = map(getImageName, labels)

        return list(zip(set(imgs), set(labels)))

    def createDSFromFiles(self, files, repeat):
        if self.give_idx:
            type = ["float32", "float32", "string", "string"]
        else:
            type = ["float32", "float32"]

        return (
            tf.data.Dataset.from_tensor_slices(files)
            .repeat(repeat)
            .map(
                lambda x: tf.py_function(self.get_sample, [x[0], x[1]], type),
                deterministic=False,
            )
        )

    def get_sample(self, image_file, label_file):
        img = np.load(image_file)
        y = np.load(label_file)

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

        files = [circle, number, text, symbol, real]
        gcd = reduce(lambda x, y: math.gcd(x, len(y)), files, self.batch_size)
        lcm = reduce(lambda x, y: x * len(y) // gcd, files, 1) * self.batch_size
        repeats = list(map(lambda x: lcm // len(x), files))

        print(
            f"""Number of {self.split} samples at '{self.data_path}': {self.num_samples}
            Number of samples collected = {list(map(len, files))}
            Dataset = [circle, number, text, symbol, real]
            repeats = {repeats}
            lcs = {lcm}
            gcd = {gcd}
            """
        )

        if seed is not None:
            np.random.seed(seed)

        datasets = list(map(lambda x: self.createDSFromFiles(*x), zip(files, repeats)))
        final_dataset = tf.data.experimental.sample_from_datasets(datasets)
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
        self.num_samples = len(glob.glob1(self.data_path, "*.png"))

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
