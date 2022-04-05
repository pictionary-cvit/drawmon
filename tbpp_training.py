"""TextBoxes++ training utils."""

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.training import (
    smooth_l1_loss,
    softmax_loss,
    focal_loss,
    ciou_loss,
    reduced_focal_loss,
    FocalRegressionLoss,
)
from ssd_training import compute_metrics

# from min_area_enclosing_box import smallest_bounding_box, enclosing_box_pca
# from box_utils.iou_rotate import iou_rotate_calculate

import tensorflow.keras.backend as K


class TBPPFocalLoss(object):
    def __init__(
        self,
        lambda_conf=1000.0,
        lambda_offsets=1.0,
        isQuads=False,
        isRbb=True,
        aabb_weight=1.0,
        rbb_weight=1.0,
        quad_weight=1.0,
        decay_factor=10.0,
        priors_xy=None,
        priors_wh=None,
        priors_variances=None,
        img_wd=512.0,
        img_ht=512.0,
        aabb_diou=True,
        rbb_diou=True,
        aabb_fr=True,
        frWithK=False,
        frWithDiou=False,
        isfl=True,
        neg_pos_ratio=3.0,
        alpha=[0.002, 0.998],
    ):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.isQuads = isQuads
        self.isRbb = isRbb

        self.alpha = alpha

        self.quad_weight = 0.0 if not self.isQuads else quad_weight
        self.aabb_weight = aabb_weight
        self.rbb_weight = rbb_weight

        self.aabb_diou = aabb_diou
        self.rbb_diou = rbb_diou

        # params for focal-regression loss (fr loss)
        self.aabb_fr = aabb_fr
        self.withK = frWithK
        self.withDiou = frWithDiou

        self.isfl = isfl
        self.neg_pos_ratio = neg_pos_ratio

        self.focalRegressionLoss = FocalRegressionLoss(gamma=0.4, image_size=(img_wd, img_ht))

        self.metric_names = [
            "loss",
            "conf_loss",
            "loc_loss",
            "loc_loss_aabb",
            "precision",
            "recall",
            "fmeasure",
            "accuracy",
            "num_pos",
            "num_neg",
            "aabb_weight",
            "rbb_weight",
        ]
        if self.isQuads:
            self.metric_names.append("loc_loss_qbb")
        if self.isRbb:
            self.metric_names.append("loc_loss_rbb")

        self.step_count = 0.0
        self.decay_factor = decay_factor

        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_variances = priors_variances

        self.variances_xy = self.priors_variances[:, 0:2]
        self.variances_wh = self.priors_variances[:, 2:4]

        self.img_wd = img_wd
        self.img_ht = img_ht

    def rbox3_to_polygon(self, rbox):

        # rbox3(x1, y1, x2, y2, h) to polygon:-
        x1, y1, x2, y2, h = rbox
        alpha = tf.math.atan2(x1 - x2, y2 - y1)
        dx = -h * tf.math.cos(alpha) / 2.0
        dy = -h * tf.math.sin(alpha) / 2.0
        xy = tf.reshape(
            tf.concat(
                [
                    x1 - dx,
                    y1 - dy,
                    x2 - dx,
                    y2 - dy,
                    x2 + dx,
                    y2 + dy,
                    x1 + dx,
                    y1 + dy,
                ],
                axis=0,
            ),
            (-1, 2),
        )

        xy = xy * (self.img_wd, self.img_ht)

        return xy

    # @tf.function
    def rbox3_to_rbox(self, rbox):

        # rbox3(x1, y1, x2, y2, h) to polygon:-
        x1, y1, x2, y2, h = rbox
        alpha = tf.math.atan2(x1 - x2, y2 - y1)
        dx = -h * tf.math.cos(alpha) / 2.0
        dy = -h * tf.math.sin(alpha) / 2.0
        xy = tf.reshape(
            tf.concat(
                [
                    x1 - dx,
                    y1 - dy,
                    x2 - dx,
                    y2 - dy,
                    x2 + dx,
                    y2 + dy,
                    x1 + dx,
                    y1 + dy,
                ],
                axis=0,
            ),
            (-1, 2),
        )

        xy = xy * (self.img_wd, self.img_ht)

        # polygon to rbox(cx, xy, w, h, theta) :-

        eps = 1e-10
        tl, tr, br, bl = xy
        # length of top and bottom edge
        dt, db = tr - tl, bl - br
        # center is mean of all 4 vetrices
        # print("xy: ", xy.shape)
        cx, cy = c = tf.math.reduce_sum(xy, axis=0) / (xy.shape[0])
        # width is mean of top and bottom edge length
        w = (tf.norm(dt) + tf.norm(db)) / 2.0
        # height is distance from center to top edge plus distance form center to bottom edge
        tlsubc = tl - c
        brsubc = br - c
        h = tf.norm(dt[0] * tlsubc[1] - dt[1] * tlsubc[0]) / (
            tf.norm(dt) + eps
        ) + tf.norm(db[0] * brsubc[1] - db[1] * brsubc[0]) / (tf.norm(db) + eps)
        # h = point_line_distance(c, tl, tr) +  point_line_distance(c, br, bl)
        # h = (norm(tl-bl) + norm(tr-br)) / 2.
        # angle is mean of top and bottom edge angle
        theta = (tf.math.atan2(dt[0], dt[1]) + tf.math.atan2(db[0], db[1])) / 2.0
        rbox = tf.concat([cx, cy, w, h, theta], axis=0)

        return rbox

    def to_box(self, offsets):
        print("Tensor: ", offsets.shape)

        # boxes = tf.zeros((offsets.shape[0], 4), dtype=offsets.dtype)
        offsets = offsets * self.priors_variances
        boxes_xy = self.priors_xy + offsets[:, 0:2] * self.priors_wh
        boxes_wh = self.priors_wh * tf.math.exp(offsets[:, 2:4])
        offsets = tf.concat(
            [boxes_xy - boxes_wh / 2.0, boxes_xy + boxes_wh / 2.0], axis=1
        )  # xmin, ymin, xmax, ymax
        offsets = tf.clip_by_value(offsets, 0.0, 1.0)

        return offsets

    def to_rbox(self, offsets):
        print("Tensor: ", offsets.shape)

        rboxs_x1y1 = (
            self.priors_xy + offsets[:, 0:2] * self.priors_wh * self.variances_xy
        )
        rboxs_x2y2 = (
            self.priors_xy + offsets[:, 2:4] * self.priors_wh * self.variances_xy
        )
        rboxs_h = tf.expand_dims(
            tf.math.exp(offsets[:, 4] * self.variances_wh[:, 1]) * self.priors_wh[:, 1],
            axis=-1,
        )
        offsets = tf.concat([rboxs_x1y1, rboxs_x2y2, rboxs_h], axis=1)

        return offsets

    def riou(self, boxes):
        pass
        """
            boxes: (10,)
            box1: (5,)
            box2: (5,)
        """
        box1 = boxes[:5]
        box2 = boxes[5:]

        radian_to_degrees = 57.2957795131

        cx, cy, h, w, theta = box1
        box1 = tf.concat([cx, cy, h, w, theta * radian_to_degrees], axis=0)

        cx, cy, h, w, theta = box2
        box2 = tf.concat([cx, cy, h, w, theta * radian_to_degrees], axis=0)

        return 1 - tf.squeeze(
            tf.squeeze(
                iou_rotate_calculate(
                    tf.expand_dims(box1, axis=0), tf.expand_dims(box2, axis=0)
                ),
                axis=0,
            ),
            axis=0,
        )

    def compute(self, y_true, y_pred, img_wd=512.0, img_ht=512.0):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)

        aabb_diou = self.aabb_diou
        rbb_diou = self.rbb_diou
        aabb_fr = self.aabb_fr

        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]

        if self.isQuads:
            num_classes = tf.shape(y_true)[2] - 17
        elif self.isRbb:
            num_classes = tf.shape(y_true)[2] - 9
        else:
            num_classes = tf.shape(y_true)[2] - 4

        eps = K.epsilon()

        # confidence loss
        if self.isQuads:
            conf_true = tf.reshape(y_true[:, :, 17:], [-1, num_classes])
            conf_pred = tf.reshape(y_pred[:, :, 17:], [-1, num_classes])
        elif self.isRbb:
            conf_true = tf.reshape(y_true[:, :, 9:], [-1, num_classes])
            conf_pred = tf.reshape(y_pred[:, :, 9:], [-1, num_classes])
        else:
            conf_true = tf.reshape(y_true[:, :, 4:], [-1, num_classes])
            conf_pred = tf.reshape(y_pred[:, :, 4:], [-1, num_classes])

        class_true = tf.argmax(conf_true, axis=1)
        class_pred = tf.argmax(conf_pred, axis=1)
        conf = tf.reduce_max(conf_pred, axis=1)

        neg_mask_float = conf_true[:, 0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos

        if self.isfl:
            print("Evaluating focal-loss......")
            conf_loss = focal_loss(conf_true, conf_pred, alpha=self.alpha)
            conf_loss = tf.reduce_sum(conf_loss)
            print(f"focal loss for classification: {conf_loss}")
            conf_loss = conf_loss / (num_total + eps)
        else:
            # softmax loss => hard negative mining
            print("Evaluating softmax-loss......")
            conf_loss = softmax_loss(conf_true, conf_pred)
            pos_conf_loss = tf.reduce_sum(conf_loss * pos_mask_float)
            pos_conf_loss = pos_conf_loss / (num_pos + eps)

            num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_neg)
            neg_conf_loss = tf.boolean_mask(conf_loss, neg_mask)
            neg_conf_loss = neg_conf_loss / (num_neg + eps)

            vals, idxs = tf.nn.top_k(neg_conf_loss, k=tf.cast(num_neg, tf.int32))
            # neg_conf_loss = tf.reduce_sum(tf.gather(neg_conf_loss, idxs))
            neg_conf_loss = tf.reduce_sum(vals)

            conf_loss = pos_conf_loss + neg_conf_loss

        # offset loss, bbox, quadrilaterals, rbbox
        if self.isQuads:
            loc_true = tf.reshape(y_true[:, :, 0:17], [-1, 17])
            loc_pred = tf.reshape(y_pred[:, :, 0:17], [-1, 17])
        elif self.isRbb:
            loc_true = tf.reshape(y_true[:, :, 0:9], [-1, 9])
            loc_pred = tf.reshape(y_pred[:, :, 0:9], [-1, 9])
        else:
            loc_true = tf.reshape(y_true[:, :, 0:4], [-1, 4])
            loc_pred = tf.reshape(y_pred[:, :, 0:4], [-1, 4])

        loc_loss = 0.0
        aabb_weight = self.aabb_weight
        rbb_weight = self.rbb_weight
        quad_weight = self.quad_weight

        # loc_loss for aabb, qbb, rbb
        if aabb_diou:
            print("Evaluating aabb-diou-loss......")
            y_true_aabb = y_true[:, :, :4]
            y_pred_aabb = y_pred[:, :, :4]

            y_true_aabb = tf.vectorized_map(self.to_box, y_true_aabb)
            y_pred_aabb = tf.vectorized_map(self.to_box, y_pred_aabb)

            y_true_aabb = tf.reshape(y_true_aabb, [-1, 4])
            y_pred_aabb = tf.reshape(y_pred_aabb, [-1, 4])

            # calculating over non-normalized
            loc_loss_aabb = ciou_loss(
                y_true_aabb * (img_wd, img_ht, img_wd, img_ht),
                y_pred_aabb * (img_wd, img_ht, img_wd, img_ht),
            )

            pos_loc_loss_aabb = tf.reduce_sum(
                loc_loss_aabb * pos_mask_float
            )  # only for positives
            loc_loss_aabb = pos_loc_loss_aabb / (num_pos + eps)
            loc_loss += aabb_weight * loc_loss_aabb
        elif aabb_fr:
            print("Evaluating aabb-focal-regression-loss......")
            y_true_aabb = y_true[:, :, :4]
            y_pred_aabb = y_pred[:, :, :4]

            y_true_aabb = tf.vectorized_map(self.to_box, y_true_aabb)
            y_pred_aabb = tf.vectorized_map(self.to_box, y_pred_aabb)

            y_true_aabb = tf.reshape(y_true_aabb, [-1, 4])
            y_pred_aabb = tf.reshape(y_pred_aabb, [-1, 4])
            # => now the boxes are un-normalized and of format (xmin, ymin, xmax, ymax)

            # calculating over non-normalized
            print(f"Image dims: {(img_wd, img_ht)}")
            loc_loss_aabb = self.focalRegressionLoss.run(
                y_true_aabb * (img_wd, img_ht, img_wd, img_ht),
                y_pred_aabb * (img_wd, img_ht, img_wd, img_ht),
                self.withK,
                self.withDiou,
            )

            pos_loc_loss_aabb = tf.reduce_sum(
                loc_loss_aabb * pos_mask_float
            )  # only for positives
            print(f"Focal Regression Loss: {pos_loc_loss_aabb}")
            loc_loss_aabb = pos_loc_loss_aabb / (num_pos + eps)
            loc_loss += aabb_weight * loc_loss_aabb
        else:
            print("Evaluating aabb-l1-loss......")
            loc_loss_aabb = smooth_l1_loss(loc_true[:, :4], loc_pred[:, :4])
            pos_loc_loss_aabb = tf.reduce_sum(
                loc_loss_aabb * pos_mask_float
            )  # only for positives
            loc_loss_aabb = pos_loc_loss_aabb / (num_pos + eps)
            loc_loss += aabb_weight * loc_loss_aabb

        if self.isQuads:
            print("Evaluating quads-l1-loss......")
            loc_loss_qbb = smooth_l1_loss(loc_true[:, 4:12], loc_pred[:, 4:12])
            pos_loc_loss_qbb = tf.reduce_sum(
                loc_loss_qbb * pos_mask_float
            )  # only for positives
            loc_loss_qbb = pos_loc_loss_qbb / (num_pos + eps)
            loc_loss += quad_weight * loc_loss_qbb

        if self.isRbb:
            # Note assuming only Rbb (otherwise it will be [12:17])
            y_true_rbb = y_true[:, :, 4:9]
            y_pred_rbb = y_pred[:, :, 4:9]

            # temp DIoU Error
            loss_rbb_diou_error = smooth_l1_loss(
                loc_true[:, 4:9], loc_pred[:, 4:9]
            )  # /10 to keep range same
            loss_rbb_diou_error = tf.boolean_mask(loss_rbb_diou_error, pos_mask)

            if rbb_diou == True:
                print("Evaluating rbb-diou-loss......")
                loss_rbb_diou_error = loss_rbb_diou_error / 10

                # offsets => real coordinates
                y_true_rbb = tf.vectorized_map(self.to_rbox, y_true_rbb)
                y_pred_rbb = tf.vectorized_map(self.to_rbox, y_pred_rbb)

                y_true_rbb = tf.reshape(y_true_rbb, [-1, 5])
                y_pred_rbb = tf.reshape(y_pred_rbb, [-1, 5])

                y_true_rbb = tf.boolean_mask(y_true_rbb, pos_mask)  # only for positives
                y_pred_rbb = tf.boolean_mask(y_pred_rbb, pos_mask)  # only for positives

                y_true_rbb_poly = tf.map_fn(
                    self.rbox3_to_polygon, y_true_rbb
                )  # (N, 4, 2)
                y_pred_rbb_poly = tf.map_fn(self.rbox3_to_polygon, y_pred_rbb)

                # y_true_rbb (N, 5)(x1, y1, x2, y2, h)(normalized) => (N, 5)(cx, cy, h, w, theta)(scaled to 512*512)
                y_true_rbb = tf.map_fn(self.rbox3_to_rbox, y_true_rbb)
                y_pred_rbb = tf.map_fn(self.rbox3_to_rbox, y_pred_rbb)

                loss_rbb_iou = tf.map_fn(
                    self.riou, tf.concat([y_true_rbb, y_pred_rbb], axis=1)
                )

                # Actual DIoU Error
                """
                fn = lambda x: tf.py_function(smallest_bounding_box, x, tf.float32)
                c2 = tf.map_fn(fn, [tf.concat([y_true_rbb_poly, y_pred_rbb_poly], axis=1)], fn_output_signature=tf.TensorSpec([], dtype=tf.float32))

                x_offset = y_true_rbb[...,0] - y_pred_rbb[..., 0]
                y_offset = y_true_rbb[...,1] - y_pred_rbb[..., 1]
                d2 = x_offset*x_offset + y_offset*y_offset

                loss_rbb_diou_error = d2/(c2 + K.epsilon())
                """

                # DIoU Error: Using PCA Approximation
                """
                w, h = enclosing_box_pca(y_true_rbb_poly, y_pred_rbb_poly)
                c2 = w*w + h*h
                x_offset = y_true_rbb[...,0] - y_pred_rbb[..., 0]
                y_offset = y_true_rbb[...,1] - y_pred_rbb[..., 1]
                d2 = x_offset*x_offset + y_offset*y_offset

                loss_rbb_diou_error = d2/(c2 + K.epsilon())
                """

                pos_loc_loss_rbb = tf.reduce_sum(loss_rbb_iou + loss_rbb_diou_error)
                loc_loss_rbb = pos_loc_loss_rbb / (num_pos + eps)
                loc_loss += rbb_weight * loc_loss_rbb
            else:
                print("Evaluating rbb-l1-loss......")
                pos_loc_loss_rbb = tf.reduce_sum(loss_rbb_diou_error)
                loc_loss_rbb = pos_loc_loss_rbb / (num_pos + eps)
                loc_loss += rbb_weight * loc_loss_rbb

        # total loss
        loss = self.lambda_conf * conf_loss + self.lambda_offsets * loc_loss

        precision, recall, accuracy, fmeasure = compute_metrics(
            class_true, class_pred, conf, top_k=100 * batch_size
        )

        print(
            "{" + " ".join(['"' + n + '": ' + n + "," for n in self.metric_names]) + "}"
        )
        return eval(
            "{" + " ".join(['"' + n + '": ' + n + "," for n in self.metric_names]) + "}"
        )

    def step(self):
        self.step_count += 1.0
