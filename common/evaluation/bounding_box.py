from enum import Enum
import sys

import numpy as np
from sklearn import metrics
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from common.utils import convert_mask_to_bbox

# For PredKind and GTKind, BBOX refers to bounding boxes already
# in a tensor format with shape (num boxes, 4).
# BBOX_DICT refers to bounding boxes in a dictionary format similar to the
# format produced byt he Matroid api. See _convert_dicts_to_bbox below
# for details
class PredKind(Enum):
    BBOX = 1
    HEATMAP = 2
    BBOX_DICT = 3

class GTKind(Enum):
    BBOX = 1
    MASK = 2
    BBOX_DICT = 3

class BoundingBoxMetrics:
    def __init__(self, image_height, image_width, **kwargs):
        self.bboxes_pred = []
        self.bboxes_gt = []
        self.confs_pred = []
        self.dices = []

        self.map = MeanAveragePrecision(**kwargs)

        self.image_height = image_height
        self.image_width = image_width


    def _convert_dicts_to_bbox(self, bbox_dicts):
        """
        Expects bbox_dicts to be in the following format:
        {
            'height': ...,
            'width': ...,
            'bboxes': {
                [{
                    'left': ...,
                    'top': ...,
                    'height': ...,
                    'width': ...,
                }, ...]
            }
        }
        """
        height, width = bbox_dicts['height'], bbox_dicts['width']

        bboxes = []
        for bbox in bbox_dicts['bboxes']:
            x1 = bbox["left"] * width
            x2 = x1 + bbox["width"] * width
            y1 = bbox["top"] * height
            y2 = y1 + bbox["height"] * height
            bboxes.append([x1, y1, x2, y2])

        if len(bboxes) == 0:
            return torch.Tensor(bboxes)

        # Perform some post-processing on the bounding boxes to
        # get them to the correct image height and width to match
        # the ground truth.
        bboxes = torch.Tensor(bboxes)
        if height < width:
            resize_ratio = self.image_width / width
            pad_height = round((self.image_height - height * resize_ratio) / 2)
            pad_width = 0
        else:
            resize_ratio = self.image_height / height
            pad_height = 0
            pad_width = round((self.image_width - width * resize_ratio) / 2)

        bboxes *= resize_ratio
        bboxes += torch.tensor([pad_width, pad_height, pad_width, pad_height])
        # Make x2 and y2 inclusive
        bboxes = torch.round(bboxes) + torch.tensor([0, 0, -1, -1])

        return bboxes

    def update(
        self,
        pred,
        gt,
        pred_kind=PredKind.BBOX, 
        gt_kind=GTKind.BBOX,
        confs_pred=None,
        pred_threshold=0,
        heatmap_conf_metric='max',
    ):
        """
        pred and gt must be of dimensionality HxW
        """
        if pred_kind == PredKind.BBOX or pred_kind == PredKind.BBOX_DICT:
            assert confs_pred is not None, "No confidences provided for bounding box annotations"
            bboxes_pred = \
                self._convert_dicts_to_bbox(pred) if pred_kind == PredKind.BBOX_DICT else torch.tensor(pred)
        elif pred_kind == PredKind.HEATMAP:
            pred = pred.squeeze()
            mask = pred >= pred_threshold
            bboxes_pred = convert_mask_to_bbox(mask)

            confs_pred = []
            for bbox in bboxes_pred:
                bbox = bbox.to(dtype=torch.int32)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if heatmap_conf_metric=='max':
                    conf = np.max(pred[y1:y2+1, x1:x2+1])
                else:
                    raise NotImplementedError(
                        f"Heatmap confidence metric of {heatmap_conf_metric} was specified but this is not supported"
                    )
                confs_pred.append(conf)
            self.confs_pred.append(confs_pred)

            # If the gt is also a mask, compute a segmentation dice score
            if gt_kind == GTKind.MASK:
                self.dices.append(metrics.f1_score(gt.flatten(), mask.flatten(), zero_division=1))

        if gt_kind == GTKind.BBOX:
            bboxes_gt = gt
        elif gt_kind == GTKind.BBOX_DICT:
            bboxes_gt = self._convert_dicts_to_bbox(gt)
        elif gt_kind == GTKind.MASK:
            bboxes_gt = convert_mask_to_bbox(gt)

        # Make x2 and y2 exclusive before storing
        if len(bboxes_pred) > 0:
            bboxes_pred += torch.tensor([0, 0, 1, 1])
        if len(bboxes_gt) > 0:
            bboxes_gt += torch.tensor([0, 0, 1, 1])

        self.bboxes_pred.append(dict(
                boxes=bboxes_pred.reshape(-1, 4),
                scores=torch.tensor(confs_pred),
                # scores=torch.tensor([0.5] * len(bboxes_pred)),
                labels=torch.tensor([0] * len(bboxes_pred)),
            ))
        self.bboxes_gt.append(dict(
                boxes=bboxes_gt,
                labels=torch.tensor([0] * len(bboxes_gt)),
            ))

    def compute(self):
        # Calculate whole-image F1 score
        whole_image_preds = [1 if len(bbox['boxes']) > 0 else 0 for bbox in self.bboxes_pred]
        whole_image_gts = [1 if len(bbox['boxes']) > 0 else 0 for bbox in self.bboxes_gt]
        f1_score = metrics.f1_score(whole_image_gts, whole_image_preds)
        # Calculate whole-image accuracy
        accuracy = np.mean(np.array(whole_image_preds) == np.array(whole_image_gts))
        # Calculate map results
        self.map.update(self.bboxes_pred, self.bboxes_gt)

        # If we've been computing dice scores for each image, compute an average
        # dice score. Note that ideally we would compute a single dice score
        # over all images, but this is space inefficient as it requires storing
        # every pixel. A benefit of computing an average dice score is that it
        # allows us to treat small and large defects relatively equally.
        if len(self.dices) > 0:
            dice = np.mean(np.array(self.dices))
        else:
            dice = np.nan

        return {
            'map_results': self.map.compute(),
            'image_classification_f1': f1_score,
            'image_classification_accuracy': accuracy,
            'segmentation dice': dice,
        }
