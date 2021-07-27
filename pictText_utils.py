import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime

from Pipeline.Core import Dataset as PDS
from Pipeline.DatasetProcessor import Transformers
from Pipeline.DatasetProcessor.Transformers.Utils import RenderingUtils
from Pipeline.DatasetProcessor.Transformers.Utils.IDUtils import TextWordId

from Pipeline.Utils.IDUtils import CircleId, MultiNumberId, SymbolId, CharacterId

class Generator(object):
    def __init__(self, data_path, padding=0):
        self.data_path = data_path
        self.padding = padding
    
    def getFileNames(self, anomaly="**", split="**", category="**", augmented = "**", svg="*.svg"):
        search_pattern = os.path.join(self.data_path, anomaly, category, split, augmented, svg)
        return glob.glob(search_pattern, recursive=True)

    def get1Channel(self, img):
        return np.array(img)[:, :, 0]

    def onlyText(self, inputs):
        sketch, metadata = inputs

        final_sketch = []
        final_metadata = []
        for i, curve in enumerate(sketch):
            if TextWordId(metadata[i]) != -1:
                final_sketch.append(curve)
                final_metadata.append(metadata[i])

        return final_sketch, final_metadata

    def stripMetadata(self, inputs):
        return inputs[0]

    def convertBboxToMinMax(self, bboxes):
        return [{
                     "bb": convertToMinMax(bbox["bb"]),
                     "obb": bbox["obb"]
                } for bbox in bboxes]

    def convertObbTo2PH(self, bboxes):
        return [{
                     "bb": bbox["bb"],
                     "obb": convertTo2PH(bbox["obb"])
                 } for bbox in bboxes]

    def flattenBboxes(self, bboxes):
        return list(zip(*[(bbox["bb"], bbox["obb"]) for bbox in bboxes]))

    def filterNonWord(self, bboxes):
        if len(bboxes[0]) == 0:
            return False

        return True

    def concatenateBbObb(self, bboxes):
        final = []
        for i in range(len(bboxes[0])):
            final.append(np.concatenate((bboxes[0][i], bboxes[1][i])))

        return final

    def startClkwiseFromTopLeft(self, bboxes):
        return [{
                "bb": bbox["bb"],
                "obb": np.roll(bbox["obb"], -1, axis=0).tolist()
                } for bbox in bboxes]

    def getOnlyObb(self, bboxes):
        return [np.roll(bbox["obb"], -1, axis=0).tolist() for bbox in bboxes]
    

    def pad(self, img):
        w, h = img.shape
        new_w = w + 2*self.padding
        new_h = h + 2*self.padding

        new_img = np.ones((new_w, new_h), dtype=img.dtype)
        new_img[:, :] = np.min(img)
        new_img[self.padding:self.padding+w, self.padding:self.padding+h] = img

        return new_img

    def shiftbox(self, bboxes):
        return [(np.array(bbox) + self.padding).tolist() for bbox in bboxes]
    
    def getClass(self, inputs):
        bbox, metadata = inputs
        return [item["anomaly_class"] for item in metadata]
    

    def getDS(self, split="train", stroke_thickness=2, erase_thickness=20, onlyTxt=False, max_erase_percentage=0.3, num_workers=1, augmented = "**", num_classes=2):
        ds = PDS.ListDataset(self.getFileNames(anomaly="anomaly", split=split, augmented=augmented))\
        .map(Transformers.SvgToPointsCallable(depth=3))\
        .map(Transformers.NormaliseSketchesCallable(min_coord=0, max_coord=512-(2*self.padding)-1))\
        
        if onlyTxt:
            ds = ds.map(self.onlyText)

        ds_img = ds.map(Transformers.RenderPointsCallable((512-(2*self.padding), 512-(2*self.padding), 3), return_imgs=True, stroke_thickness=stroke_thickness, erase_thickness=erase_thickness))\
        .map(self.get1Channel)
        
        if self.padding > 0: ds_img = ds_img.map(self.pad)

        idFns = [TextWordId, MultiNumberId, SymbolId, CircleId]

	if num_classes=2:

            ds_bbox = ds\
            .map(Transformers.AbsPointsToOBBCallable(TextWordId, max_erase_percentage=max_erase_percentage, num_workers=num_workers))\
            .map(self.stripMetadata)\
            .map(self.convertBboxToMinMax)\
            .map(self.startClkwiseFromTopLeft)\
            .map(self.getOnlyObb)
        else:
            # multiclass
            ds_bbox_class = ds\
            .map(Transformers.AbsPointsToOBBCallable(idFns, max_erase_percentage=max_erase_percentage, num_workers=num_workers))\
        
            ds_only_bbox = ds_bbox_class\
            .map(self.stripMetadata)\
            .map(self.convertBboxToMinMax)\
            .map(self.startClkwiseFromTopLeft)\
            .map(self.getOnlyObb)

            ds_only_class = ds_bbox_class\
            .map(self.getClass)

            ds_bbox = ds_only_bbox.zip(ds_only_class)
        
        if self.padding > 0: ds_bbox = ds_bbox.map(self.shiftbox)
    
        ds = ds_img.zip(ds_bbox)
        print(len(ds))
        
        return ds

def convertToCentroid(bb):
    """
        bb: (4 coords = 8 values)

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

def convertToMinMax(bb):
    """
        bb: (4 coords = 8 values)

        return mx, my, Mx, My
    """
    bb = np.array(bb)

    mx, my = np.min(bb, axis=0)
    Mx, My = np.max(bb, axis=0)

    return np.array((mx, my, Mx, My))

def convertCentroidToMinMax(bb):
    """
        bb (cx, cy, w, h)

        return (mx, my, Mx, My)
    """
    bb = np.array(bb)
    if len(bb.shape) == 1:
        cx, cy, w, h = bb
        return np.array((cx-w/2, cy-h/2, cx+w/2, cy+h/2))

    cx = bb[:, 0:1]
    cy = bb[:, 1:2]
    w = bb[:, 2:3]
    h = bb[:, 3:4]

    return np.concatenate(
        ( cx-w/2, cy-h/2, cx+w/2, cy+h/2 ),
        axis = 1
    )

def convertMinMaxToCenteroid(bb):
    """
        bb (mx, my, Mx, My)

        return (cx, cy, w, h)
    """
    bb = np.array(bb)

    if len(bb.shape) == 1:
        mx, my, Mx, My = bb
        return np.array(((mx+Mx)/2, (my+My)/2, Mx - mx, My-my))

    mx = bb[:, 0:1]
    my = bb[:, 1:2]
    Mx = bb[:, 2:3]
    My = bb[:, 3:]

    return np.concatenate(
        ( (mx+Mx)/2, (my+My)/2, Mx - mx, My-my ),
        axis = 1
    )

def convertTo2PH(obb):
    """
        obb: (4 coords = 8 values)

        return mx, my, Mx, My, h
    """
    obb = np.array(obb)

    my = np.min(obb[:,1], axis=0)

    mx = np.max(obb)
    bottom_left_point = -1

    for i, point in enumerate(obb):
        if point[1] == my and point[0] <= mx:
            mx = point[0]
            bottom_left_point = i

    h = np.sqrt(np.sum((obb[bottom_left_point]-obb[bottom_left_point-1])**2))
    return np.array((*obb[bottom_left_point], *obb[bottom_left_point-2], h))
