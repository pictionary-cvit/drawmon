"""Some utils for TextBoxes++."""

from unittest import result
import numpy as np
import matplotlib.pyplot as plt

from ssd_utils import PriorUtil as SSDPriorUtil
from ssd_utils import iou, non_maximum_suppression, non_maximum_suppression_slow
from utils.bboxes import polygon_to_rbox3, rbox3_to_polygon
from utils.vis import plot_box


class PriorUtil(SSDPriorUtil):
    """Utility for SSD prior boxes.
       AABB: Axis-Aligned Bounding Box
       RBB: Rotated Bounding Box
    """

    def encode(self, gt_data, overlap_threshold=0.5, debug=False, quads=False, num_classes=2):
        # calculation is done with normalized sizes
        
        # TODO: empty ground truth
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)
        
        num_classes = num_classes
        num_priors = self.priors.shape[0] # number of priors
        
        gt_polygons = np.copy(gt_data[:,:8]) # normalized quadrilaterals
        gt_rboxes = np.array([polygon_to_rbox3(np.reshape(p, (-1,2))) for p in gt_data[:,:8]]) # normalized rbox for each quad
        
        # minimum horizontal bounding rectangles
        gt_xmin = np.min(gt_data[:,0:8:2], axis=1)
        gt_ymin = np.min(gt_data[:,1:8:2], axis=1)
        gt_xmax = np.max(gt_data[:,0:8:2], axis=1)
        gt_ymax = np.max(gt_data[:,1:8:2], axis=1)
        gt_boxes = self.gt_boxes = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax]).T # normalized xmin, ymin, xmax, ymax
        
        # give classes: 0: background, 1: text=> text-class [0, 1], bg class [1, 0]
        gt_class_idx = np.asarray(gt_data[:,-1]+0.5, dtype=np.int)
        gt_one_hot = np.zeros([len(gt_class_idx),num_classes])
        gt_one_hot[range(len(gt_one_hot)),gt_class_idx] = 1 # one_hot classes including background
        
        ## NOTICE: WE WILL NOW USE CALCULATED AABB(for each gt) FOR IOU CALCULATION AND ASSIGNING GT TO PRIORS
        
        # iou of each gt_box with each prior (Note: Transpose) => returns (num_priors, num_boxes)
        gt_iou = np.array([iou(b, self.priors_norm) for b in gt_boxes]).T
        
        # assigne gt to priors
        max_idxs = np.argmax(gt_iou, axis=1) # id of gt that has max-iou for each prior, shape: (num_priors) 
        max_val = gt_iou[np.arange(num_priors), max_idxs] # max-iou for each prior, shape: (num_priors)
        prior_mask = max_val > overlap_threshold 
        match_indices = max_idxs[prior_mask] # indices of gt that has been assigned to some prior

        # (index of prior, index of gt it overlaps with(having iou > threshold))
        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices))) 

        # set prior labels 
        confidence = np.zeros((num_priors, num_classes))
        confidence[:,0] = 1 # set every-prior to bg class
        # set the priors(which matched to some gt with iou>thres) to class of that gt
        confidence[prior_mask] = gt_one_hot[match_indices] 

        gt_xy = (gt_boxes[:,2:4] + gt_boxes[:,0:2]) / 2. # coordinates of center of aabb:gt_boxes
        gt_wh = gt_boxes[:,2:4] - gt_boxes[:,0:2] # width and ht of aabb:gt_boxes
        gt_xy = gt_xy[match_indices] # get center of only aabb:gt's that has been assigned to some prior
        gt_wh = gt_wh[match_indices]
        gt_polygons = gt_polygons[match_indices] # get quad corresponding to aabb:gt's that has been assigned to some prior
        gt_rboxes = gt_rboxes[match_indices]
        
        priors_xy = self.priors_xy[prior_mask] / self.image_size 
        priors_wh = self.priors_wh[prior_mask] / self.image_size
        variances_xy = self.priors_variances[prior_mask,0:2]
        variances_wh = self.priors_variances[prior_mask,2:4]
        
        # compute AABB local offsets for each prior with class[0, 1] (=> that has been assigned to some gt)
        offsets = np.zeros((num_priors, 4)) # init offeset for each prior with [0, 0, 0, 0]
        offsets[prior_mask,0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask,2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask,0:2] /= variances_xy
        offsets[prior_mask,2:4] /= variances_wh
        
        # compute QUADS local offsets
        offsets_quads = np.zeros((num_priors, 8))
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2]) # gives coords of priors as (xmin, ymin, xmax, ymax)
        #ref = np.tile(priors_xy, (1,4))
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points => 1st corner(xmin, ymin) i.e (0,1) of priors_xy_minmax, 2nd corner (xmax, ymin) i.e (2, 1) 
        offsets_quads[prior_mask,:] = (gt_polygons - ref) / np.tile(priors_wh, (1,4)) / np.tile(variances_xy, (1,4))
        
        # compute RBB local offsets 
        offsets_rboxs = np.zeros((num_priors, 5))
        offsets_rboxs[prior_mask,0:2] = (gt_rboxes[:,0:2] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,2:4] = (gt_rboxes[:,2:4] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,4] = np.log(gt_rboxes[:,4] / priors_wh[:,1]) / variances_wh[:,1]
        
        if self.isQuads and self.isRbb:
            return np.concatenate([offsets, offsets_quads, offsets_rboxs, confidence], axis=1)
        elif self.isRbb:
            return np.concatenate([offsets, offsets_rboxs, confidence], axis=1)
        elif self.isQuads:
            return np.concatenate([offsets, offsets_quads, confidence], axis=1)
        else:
            return np.concatenate([offsets, confidence], axis=1)
        
    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=True, sparse=True, quads=False, class_idx = -1):
        '''
        # calculation is done with normalized sizes
        INPUT:    
            # mbox_loc, mbox_rbox, mbox_conf
            # 4,5,2
        OUTPUT:    
            # boxes, rboxes, confs scores, labels (Note: scores Not probability)
            # 4,5,1,1
        '''
        try:
            model_output = model_output.numpy()
        except:
            pass        

        if self.isQuads and self.isRbb:
            prior_mask = model_output[:,17:] > confidence_threshold
        elif self.isRbb:
            prior_mask = model_output[:,9:] > confidence_threshold
        elif self.isQuads:
            prior_mask = model_output[:,12:] > confidence_threshold
        else:
            prior_mask = model_output[:,4:] > confidence_threshold
            
        
        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            # print(f"model_output.shape: {model_output.shape, model_output.dtype}")
            # print(f"prior_mask.shape: {prior_mask.shape, prior_mask.dtype}")
            mask = np.any(prior_mask[:,1:], axis=1)
            # print(f"mask.shape: {mask.shape, mask.dtype}")
            prior_mask = prior_mask[mask]
            # print(f"prior_mask.shape: {prior_mask.shape, prior_mask.dtype}")
            mask = np.ix_(mask)[0]
            # print(f"mask.shape: {mask.shape, mask.dtype}")
            mask = list(mask.astype(int))
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors_variances[mask,:]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors_variances
        
        
        # NOTE: Below calculations are only for masked priors_xy... and masked model_output
        
        offsets = model_output[:,:4]
        if self.isQuads and self.isRbb:
            offsets_quads = model_output[:,4:12]
            offsets_rboxs = model_output[:,12:17]
            confidence = model_output[:,17:]
        elif self.isRbb:
            offsets_rboxs = model_output[:,4:9]
            confidence = model_output[:,9:]
        elif self.isQuads:
            offsets_quads = model_output[:,4:12]
            confidence = model_output[:,12:]
        else:
            confidence = model_output[:,4:]
        
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2]) # gives coords of priors as (xmin, ymin, xmax, ymax)
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points => 1st corner(xmin, ymin) i.e (0,1) of priors_xy_minmax, 2nd corner (xmax, ymin) i.e (2, 1) 
        variances_xy = priors_variances[:,0:2]
        variances_wh = priors_variances[:,2:4]
        
        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:,0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:,2:4])
        boxes[:,0:2] = boxes_xy - boxes_wh / 2. # xmin, ymin
        boxes[:,2:4] = boxes_xy + boxes_wh / 2. # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)
        
        # do non maximum suppression
        results = []
        if class_idx == -1:
            class_iter = range(1, num_classes)
        else:
            class_iter = [class_idx]

        for c in class_iter: # No bg class
            mask = prior_mask[:,c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]
                
                if fast_nms:
                    idx = non_maximum_suppression(
                            boxes_to_process, confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                            boxes_to_process, confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                
                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx),1)) * c
                
                if self.isQuads:
                    good_quads = ref[mask][idx] + offsets_quads[mask][idx] * np.tile(priors_wh[mask][idx] * variances_xy[mask][idx], (1,4))
                if self.isRbb:
                    good_rboxs = np.empty((len(idx), 5))
                    good_rboxs[:,0:2] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,0:2] * priors_wh[mask][idx] * variances_xy[mask][idx]
                    good_rboxs[:,2:4] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,2:4] * priors_wh[mask][idx] * variances_xy[mask][idx]
                    good_rboxs[:,4] = np.exp(offsets_rboxs[mask][idx,4] * variances_wh[mask][idx,1]) * priors_wh[mask][idx,1]
                
                if self.isQuads and self.isRbb:
                    c_pred = np.concatenate((good_boxes, good_quads, good_rboxs, good_confs, labels), axis=1)
                elif self.isRbb:
                    c_pred = np.concatenate((good_boxes, good_rboxs, good_confs, labels), axis=1)
                elif self.isQuads:
                    c_pred = np.concatenate((good_boxes, good_quads, good_confs, labels), axis=1)
                else:
                    c_pred = np.concatenate((good_boxes, good_confs, labels), axis=1)
                    
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            if self.isQuads and self.isRbb:
                order = np.argsort(-results[:, 17])
            elif self.isRbb:
                order = np.argsort(-results[:, 9])
            elif self.isQuads:
                order = np.argsort(-results[:, 12])
            else:
                order = np.argsort(-results[:, 4])
                
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0,6))
        self.results = results

        # results of format for an image: (number_of_boxes_in_that_image, 5)
        # 4 is (xmin, ymin, xmax, ymax), left 1 is score of that box
        # multiply each bbox as: bbox*(w,h,w,h), box_format is 'xyxy'
        '''
        Todo: Check overlap(iou) of each box with each box
        if overlap is greater then iou_thres, then merge those boxes
        '''
        self.merge_overlapping_boxes(results)
        return results

    def merge_box(self, box1, box2):
        return (min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[0], box2[0]), max(box1[0], box2[0]))

    def merge_overlapping_boxes(self, results):
        # results of format for an image: (number_of_boxes_in_that_image, 5)
        # 4 is (xmin, ymin, xmax, ymax), left 1 is score of that box
        # multiply each bbox as: bbox*(w,h,w,h), box_format is 'xyxy'
        if len(results) <= 1:
            return results
        for i in range(len(results)-1):
            for j in range(len(results)):
                if (results[i][0:4] == (-1000, -1000, -1000, -1000)).any():
                    break
                if (results[j][0:4] == (-1000, -1000, -1000, -1000)).any():
                    continue
                overlaps = iou(results[i][0:4], np.array([results[j][0:4]]))[0]
                if overlaps > self.iou_merge_thres:
                    box = self.merge_box(results[i][0:4], results[j][0:4])
                    results[i][0:4] = box
                    results[j][0:4] = (-1000, -1000, -1000, -1000)
        
        final_boxes = []
        for i in range(len(results)):
            if (results[i][0:4] != (-1000, -1000, -1000, -1000)).all():
                final_boxes.append(results[i])
        
        return final_boxes


    def plot_results(self, results=None, classes=None, show_labels=False, gt_data=None, gt_data_decoded=None, confidence_threshold=None, quads=False, hw = None, pad=0):
        if results is None:
            results = self.results
        if confidence_threshold is not None and results is not None and len(results) > 0:
            if self.isQuads and self.isRbb:
                mask = results[:, 17] > confidence_threshold
            elif self.isRbb:
                mask = results[:, 9] > confidence_threshold
            elif self.isQuads:
                mask = results[:, 12] > confidence_threshold
            else:
                mask = results[:, 4] > confidence_threshold
                
                
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes)+1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        if hw is None:
            h, w = im.get_size()
        else:
            h, w = hw

        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0]+1
                color = 'g' if classes == None else colors[label]
                xy = np.reshape(box[:8], (-1,2)) * (w,h)
                ax.add_patch(plt.Polygon(xy, fill=True, color=color, linewidth=3, alpha=0.3))
        
        if gt_data_decoded is not None:
            for r in gt_data_decoded:
                bbox = r[0:4]
                if self.isQuads:
                    quad = r[4:12]
                    plot_box(np.reshape(quad,(-1,2))*(w,h), box_format='polygon', color='y')
                elif self.isRbb:
                    rbox = r[4:9]
                    plot_box(rbox3_to_polygon(rbox)*(w,h), box_format='polygon', color='y')
                else:
                    plot_box(bbox*(w,h,w,h), box_format='xyxy', color='y')

        # draw prediction
        for r in results:
            bbox = r[0:4]
            if self.isQuads:
                quad = r[4:12]
                rbox = r[12:17]
                confidence = r[17]
                label = int(r[18])
            elif self.isRbb:
                rbox = r[4:9]
                confidence = r[9]
                label = int(r[10])
            else:
                confidence = r[4]
                label = int(r[5])

            plot_box(bbox*(w,h,w,h)+pad, box_format='xyxy', color=colors[label])
            if self.isQuads:
                plot_box(np.reshape(quad,(-1,2))*(w,h), box_format='polygon', color='r')
            
            if self.isRbb:
                plot_box(rbox3_to_polygon(rbox)*(w,h), box_format='polygon', color='g')
                plt.plot(rbox[[0,2]]*(w,w), rbox[[1,3]]*(h,h), 'oc', markersize=4)
                
            if show_labels:
                label_name = label if classes == None else classes[label]
                color = 'r' if classes == None else colors[label]
                xmin, ymin = bbox[:2]*(w,h)
                display_txt = '%0.2f, %s' % (confidence, label_name)
                ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

