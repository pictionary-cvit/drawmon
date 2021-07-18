from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pictText_utils import convertMinMaxToCenteroid
import numpy as np
import sys
from io import StringIO 

class PycocoMetric:
    def __init__(self, iou_threshold, confidence_threshold, top_k, num_classes, show_summary=False):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.num_classes = num_classes
        self.show_summary = show_summary
    
    def __call__(self, y_true, y_pred):
        
        '''
            y_true, y_pred must be in decoded format
            format: # boxes, quad, rboxes, confs, labels
                    # 4,8,5,1,1
             
             # boxes format: xmin, ymin, xmax, ymax
        '''
        
        dataset = {
                'images': [],
                'annotations': [],
                'categories': []
            }


        counter= 0
        annotation_id = 1
        image_id = 1
        for boxes in y_true:
            if (len(boxes) == 0): print(image_id)
            for box in boxes:
                counter+=1
                print(box, image_id)
                area = abs((box[2] - box[0]) * (box[3] - box[1]))
                dataset['annotations'].append({
                              'id': annotation_id,
                              'image_id': image_id,
                              'category_id': int(box[-1]),
                              'bbox': convertMinMaxToCenteroid(box[:4]).tolist(),
                              'area': area,
                              'iscrowd': 0
                          })
            
                annotation_id+=1
            dataset['images'].append({'id': int(image_id)})
            image_id+=1
            
        category_ids = [i for i in range(self.num_classes)]
        
        dataset['categories'] = [
          {'id': int(category_id)} for category_id in category_ids
        ]
        
        detections = []
        pred_img_id = 1
        for boxes in y_pred:
            
            for box in boxes:
                x,y,w,h = convertMinMaxToCenteroid(box[:4])
                final_box = [pred_img_id, x, y, w, h, box[-2], box[-1]] # format imageID, x1, y1, w, h, score, class
                detections.append(final_box)
            
            pred_img_id+=1
        detections = np.array(detections)
        
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
        
        if self.show_summary:
            coco_eval.summarize()
            
        names = ['='.join(item.split("=")[:-1]) for item in mystdout.getvalue().split("\n")]
        coco_metrics = list(zip(names, coco_eval.stats))
        
        return coco_metrics
