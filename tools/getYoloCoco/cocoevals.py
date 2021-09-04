from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from pictText_utils import convertMinMaxToCenteroid
import numpy as np
import sys
from io import StringIO 


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


class PycocoMetric:
    def __init__(self):
        pass
    
    def __call__(self, dataset, detections):   
        
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        image_ids = list(set(detections[:, 0]))
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        coco_eval.summarize()

        sys.stdout = old_stdout
        
        names = ['='.join(item.split("=")[:-1]) for item in mystdout.getvalue().split("\n")]
        coco_metrics = list(zip(names, coco_eval.stats))
        
        print(coco_metrics)
        return coco_metrics


