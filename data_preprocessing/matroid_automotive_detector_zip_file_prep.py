import csv
import os
import shutil
import sys

sys.path.insert(0, '..')

import torch
from tqdm import tqdm

from common.datasets import custom_dataset, utils
from common.utils import convert_bbox_to_dicts, convert_mask_to_bbox

dataset = custom_dataset.CustomDataset(
    "../datasets/automotive/processed", classname="top_easy_split", resize=1024, split=utils.DatasetSplit.TEST, patches_per_row=1, patches_per_column=1
)
dataset_2 = custom_dataset.CustomDataset(
    "../datasets/automotive/processed", classname="top_easy_split", resize=1024, split=utils.DatasetSplit.TRAIN, patches_per_row=1, patches_per_column=1
)

# matroid_train_set = [
#     '210318_130321_28_scratch',
#     '210318_130109_26_gas_mark',
#     '210318_124548_13_flash_chrome_burn',
#     '210318_125625_22_gas_mark',
#     '210318_124933_17_pit'
# ]
# matroid_train_set = ['210318_123651_02_pit',
#  '210318_125717_23_scratch',
#  '210318_123551_01_contamination',
#  '210318_125625_22_gas_mark',
#  '210318_124054_06_scratch']
# matroid_train_set = ['210318_132627_49_skip_plate',
#  '210318_132705_50_pit',
#  '210318_131620_42_scratch',
#  '210318_124011_05_contamination',
#  '210318_125758_24_splay']
matroid_train_set = ['210318_130853_34_scratch_o_splay',
 '210318_132752_51_skip_plate',
 '210318_132548_48_skip_plate',
 '210318_124714_15_scratch',
 '210318_125758_24_splay']

all_bbox_dicts = {}
for example in tqdm(dataset):
    image, mask, image_name, image_path = (
        example["image"], example["mask"], example["image_name"], example["image_path"]
    )
    # print(image_name)
    skip=True
    for name in matroid_train_set:
        if name in image_name:
            # print(name)
            skip=False

    if skip:
        continue
    print(image_name)

    image_name = image_name[:-2] # Remove the index
    _, height, width = image.shape
    bbox_dicts = convert_bbox_to_dicts(convert_mask_to_bbox(mask.numpy()), height, width)

    all_bbox_dicts[(image_name, image_path)] = bbox_dicts
for example in tqdm(dataset_2):
    image, mask, image_name, image_path = (
        example["image"], example["mask"], example["image_name"], example["image_path"]
    )
    # print(image_name)
    skip=True
    for name in matroid_train_set:
        if name in image_name:
            skip=False

    if skip:
        continue
    print(image_name)

    image_name = image_name[:-2] # Remove the index
    _, height, width = image.shape
    bbox_dicts = convert_bbox_to_dicts(convert_mask_to_bbox(mask.numpy()), height, width)

    all_bbox_dicts[(image_name, image_path)] = bbox_dicts


with open(
    '../datasets/automotive/matroid_detector_data_file/bbox.csv', 'w', newline=''
) as csvfile:
    writer = csv.writer(csvfile)
    for (image_name, image_path), bbox_dicts in all_bbox_dicts.items():
        image_copy_dir = '../datasets/automotive/matroid_detector_data_file/defects'
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
                    'defects', 
                    'positive', 
                    image_name
                ]
            )
