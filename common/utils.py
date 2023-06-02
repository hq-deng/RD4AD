import csv
import os
import os.path

import cv2
import numpy as np
from numpy import ndarray
import torch
from torchvision.ops import masks_to_boxes

# Taken from CFLOW implementation
class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()
        
        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))

# Adapted from PatchCore implementation
def compute_and_store_results(
    results,
    column_names,
    run_save_path,
):
    savename = f'{run_save_path}.csv'
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        csv_writer.writerow(header)
        csv_writer.writerow(results)

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def convert_bbox_to_dicts(bboxes, height, width):
    # Converts an NxHxW tensor of bounding boxes
    # to the Matroid format for bounding box dictionaries.
    bbox_dicts = []
    for bbox in bboxes:
        # Assumes x2 and y2 are INCLUSIVE of the bounding box
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_dict = {
            'left': x1 / width,
            'top': y1 / height,
            'width': (x2 - x1 + 1) / width,
            'height': (y2 - y1 + 1) / height,
        }
        for key in bbox_dict:
            bbox_dict[key] = bbox_dict[key].item() # Convert from tensor
        bbox_dicts.append(bbox_dict)

    return bbox_dicts

def convert_mask_to_bbox(mask: ndarray):
    # First split the mask into its connected components to convert to
    # the correct form for masks_to_boxes.
    mask = np.squeeze(mask).astype(np.int8)

    num_labels, label_ids, _, _ = cv2.connectedComponentsWithStats(mask)
    split_mask = []
    for i in range(1, num_labels):
        split_mask.append(label_ids == i)
    split_mask = torch.Tensor(np.array(split_mask))
    # Note that the coordinates of the lower righthand corner of the box
    # are INCLUSIVE of the actual size of the bounding box.
    bboxes = masks_to_boxes(split_mask)
    return bboxes
