import csv
import os
import shutil
import sys

sys.path.insert(0, '..')

import torch
from tqdm import tqdm

from common.datasets import bsd_dataset, utils
from common.utils import convert_bbox_to_dicts, convert_mask_to_bbox

dataset = bsd_dataset.BSDDataset(
    "../datasets/BSData", resize=1024, split=utils.DatasetSplit.MATROID_TRAIN, no_square_pad=True, patches_per_row=1, patches_per_column=1, train_split_percent=0.75
)

all_bbox_dicts = {}
for example in tqdm(dataset):
    image, mask, image_name, image_path = (
        example["image"], example["mask"], example["image_name"], example["image_path"]
    )

    image_name = image_name[:-2] # Remove the index
    _, height, width = image.shape
    bbox_dicts = convert_bbox_to_dicts(convert_mask_to_bbox(mask.numpy()), height, width)

    all_bbox_dicts[(image_name, image_path)] = bbox_dicts

with open(
    '../datasets/BSData/matroid_detector_v2_data_file/bbox.csv', 'w', newline=''
) as csvfile:
    writer = csv.writer(csvfile)
    for (image_name, image_path), bbox_dicts in all_bbox_dicts.items():
        image_copy_dir = '../datasets/BSData/matroid_detector_v2_data_file/pitting'
        os.makedirs(image_copy_dir, exist_ok=True)

        shutil.copy(image_path, image_copy_dir)

        for bbox_dict in bbox_dicts:
            left, top = bbox_dict['left'], bbox_dict['top']
            width, height = bbox_dict['width'], bbox_dict['height']
            right, bottom = left + width, top + height

            writer.writerow(
                [
                    round(left, 3), 
                    round(top, 3), 
                    round(right, 3), 
                    round(bottom, 3), 
                    'pitting', 
                    'positive', 
                    image_name
                ]
            )
