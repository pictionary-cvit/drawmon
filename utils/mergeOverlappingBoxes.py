import tensorflow as tf
import numpy as np


class MergeOverlappingBoxes(object):
    """docstring for mergeOverlappingBoxes"""
    def __init__(self, overlapping_thres, image_size=(512,512)):
        super(MergeOverlappingBoxes, self).__init__()
        self.overlapping_thres = overlapping_thres
        self.image_size = image_size

    def merge_box(self, box1, box2):
        # box is unnormalized of type (x,y,x,y)
        return (min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[0], box2[0]), max(box1[0], box2[0]))

    def find_area(self, box):
        # box is unnormalized of type (x,y,x,y)
        return abs((box[2]-box[0])*(box[3]-box[1]))

    def find_overlapping_area(self, box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        xA = max(x11, x21)
        yA = max(y11, y21)
        xB = min(x12, x22)
        yB = min(y12, y22)

        interArea = max((xB - xA), 0) * max((yB - yA), 0)

        return interArea

    def run(self, bboxes):
        # bboxes of format for an image: (number_of_boxes_in_that_image, 5)
        # 4 is (xmin, ymin, xmax, ymax), left 1 is score of that box
        # multiply each bbox as: bbox*(w,h,w,h), box_format is 'xyxy'
        wd = self.image_size[0]
        ht = self.image_size[1]

        if len(bboxes) <= 1:
            return bboxes
        for i in range(len(bboxes)-1):
            for j in range(i+1, len(bboxes)):
                if (bboxes[i][0:4] == (-1000, -1000, -1000, -1000)).any():
                    break
                if (bboxes[j][0:4] == (-1000, -1000, -1000, -1000)).any():
                    continue
                unnormalized_box1 = bboxes[i][0:4]*(wd, ht, wd, ht) 
                unnormalized_box2 = bboxes[j][0:4]*(wd, ht, wd, ht)
                
                area1 = self.find_area(unnormalized_box1)
                area2 = self.find_area(unnormalized_box2)

                # overlaps = iou(unnormalized_box1, np.array([unnormalized_box2]))[0]
                overlapping_area = self.find_overlapping_area(unnormalized_box1, unnormalized_box2)
                
                is_lies_inside = (overlapping_area/area1 >= self.overlapping_thres) or (overlapping_area/area2 >= self.overlapping_thres)
                
                # if overlaps > self.iou_merge_thres or is_lies_inside:
                if is_lies_inside:
                    box = self.merge_box(bboxes[i][0:4], bboxes[j][0:4])
                    bboxes[i][0:4] = box
                    bboxes[j][0:4] = (-1000, -1000, -1000, -1000)
        
        final_boxes = []
        for i in range(len(bboxes)):
            if (bboxes[i][0:4] != (-1000, -1000, -1000, -1000)).all():
                final_boxes.append(bboxes[i])
        
        return np.array(final_boxes)

        
obj = MergeOverlappingBoxes(overlapping_thres=0.8)

bboxes = np.array([[0.0, 0.0, 2.0, 2.0, 1], [1.0, 1.0, 2.0, 2.0, 1]])
print(obj.run(bboxes))