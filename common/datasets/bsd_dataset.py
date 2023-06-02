import json
import os
from enum import Enum

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

from common.datasets.utils import DatasetSplit, SquarePad

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SPLITS_FILENAME = "splits.json"

class BSDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ball screw drive open-source dataset.
    """

    def __init__(
        self,
        source,
        resize=256,
        split=DatasetSplit.TRAIN,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the BSD data folder.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. bsd_dataset.DatasetSplit.TRAIN. Note that
                   bsd_dataset.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split

        self.patches_per_image = kwargs.get('patches_per_image', None)
        self.patches_per_row = kwargs.get('patches_per_row', self.patches_per_image)
        assert(self.patches_per_row is not None)
        self.patches_per_column = kwargs.get('patches_per_column', 1)
        if self.patches_per_image is None:
            self.patches_per_image = self.patches_per_row * self.patches_per_column

        self.train_split_percent = kwargs.get('train_split_percent', 1)

        self.transform_img = [
            SquarePad(),
            transforms.Resize(resize),
            # transforms.CenterCrop((resize, resize * 2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_mask = [
            SquarePad(),
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            # Open the mask as black and white, since there may be different colors
            # assigned to different semantic classes of defects.
            # transforms.CenterCrop((resize, resize * 2)),
            transforms.Lambda(lambda x : x.convert('L')),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : (x != 0).to(dtype=x.dtype)),
        ]

        if kwargs.get('no_square_pad'):
            self.transform_img = self.transform_img[1:]
            self.transform_mask = self.transform_mask[1:]

        crop_width = kwargs.get('crop_width')
        crop_height = kwargs.get('crop_height')
        if crop_width and crop_height:
            self.transform_img.insert(2, transforms.CenterCrop((crop_height, crop_width)))
            self.transform_mask.insert(2, transforms.CenterCrop((crop_height, crop_width)))
        else:
            assert not (crop_width or crop_height), "Only one of crop_width and crop_height provided."

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.data_to_iterate = self.get_image_data()

        # Since anomaly detection models tend to perform best on larger defects, this allows
        # us to remove examples where the total number of defective pixels is very small.
        if kwargs.get('filter_big_defects', False):
            self.data_to_iterate = list(filter(self._filter_big_defects, self.data_to_iterate))

    def __getitem__(self, idx):
        image_path, mask_path, index = self.data_to_iterate[idx]
        # Not necessary to convert to 'RGB' unless if the image is
        # actually B&W
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask = torch.zeros([1, *image.size()[1:]])
        if ((self.split == DatasetSplit.TEST or self.split == DatasetSplit.MATROID_TRAIN) and
            mask_path is not None):
            mask = np.zeros_like(cv2.imread(image_path))
            with open(mask_path) as f:
                mask_info = json.load(f)["shapes"]
                points = [np.array(shape["points"], dtype=np.int32) for shape in mask_info]
                cv2.fillPoly(mask, pts=points, color=(255, 255, 255))
                mask = self.transform_mask(PIL.Image.fromarray(mask))

        width = mask.size()[-1]
        height = mask.size()[-2]

        patch_width = width // self.patches_per_row
        patch_height = height // self.patches_per_column

        starting_x = patch_width * (index % self.patches_per_row)
        starting_y = patch_height * (index // self.patches_per_row)

        image = \
            image[:, starting_y:starting_y+patch_height, starting_x:starting_x+patch_width]
        mask = \
            mask[:, starting_y:starting_y+patch_height, starting_x:starting_x+patch_width]

        is_anomaly = (mask.sum() > 0).item()

        return {
            "image": image,
            "mask": mask,
            "classname": 'bsd',
            "is_anomaly": (mask.sum() > 0).item(),
            "image_name": f"{image_path.split('/')[-1]}_{index}",
            "image_path": image_path,
        }

    def unpatch(self, gt_patches, img_patches, pred_patches):
        """
        Requires these patches to have been collected with a batch size of 1, where
        the leading dimension of each patch is the batch size.
        """
        assert len(gt_patches) == len(img_patches) and len(gt_patches) == len(pred_patches), \
            "Attempting to unpatch image, gt, and patches, but there are different lengths"
        assert len(gt_patches) <= self.patches_per_image, \
            "More patches provided to unpatch function than there should be per image"

        if len(gt_patches) < self.patches_per_image:
            # Not ready to unpatch
            return False, None, None, None
        else:
            assert len(gt_patches[0].shape) == 4 and gt_patches[0].shape[0] == 1, \
                f"Wrong shape for gt_patch. Expected (1, 1, H, W) but got {gt_patches[0].shape}"

            gt = torch.zeros_like(gt_patches[0]) # By this point, there must be at least 1 patch
            img = torch.zeros_like(img_patches[0])
            pred = torch.zeros_like(pred_patches[0])

            gt = gt.repeat(1, 1, self.patches_per_column, self.patches_per_row)
            img = img.repeat(1, 1, self.patches_per_column, self.patches_per_row)
            pred = pred.repeat(self.patches_per_column, self.patches_per_row)

            _, _, patch_height, patch_width = gt_patches[0].shape

            for index in range(len(gt_patches)):
                starting_x = patch_width * (index % self.patches_per_row)
                starting_y = patch_height * (index // self.patches_per_row)
                img[:, :, starting_y:starting_y+patch_height, starting_x:starting_x+patch_width] = \
                    img_patches[index]
                gt[:, :, starting_y:starting_y+patch_height, starting_x:starting_x+patch_width] = \
                    gt_patches[index]
                pred[starting_y:starting_y+patch_height, starting_x:starting_x+patch_width] = \
                    pred_patches[index]

            return True, gt, img, pred

    def __len__(self):
        return len(self.data_to_iterate)

    def _filter_big_defects(self, img_tuple):
        THRESHOLD = 500
        gt = img_tuple[1]
        index = img_tuple[2]
        if gt is None:
            return True

        gt_arr = self.transform_mask(PIL.Image.open(gt))

        width = gt_arr.size()[-1]
        height = gt_arr.size()[-2]

        patch_width = width // self.patches_per_row
        patch_height = height // self.patches_per_column

        starting_x = patch_width * (index % self.patches_per_row)
        starting_y = patch_height * (index // self.patches_per_row)

        gt_arr = \
            gt_arr[:, starting_y:starting_y+patch_height, starting_x:starting_x+patch_width]

        return torch.sum(gt_arr) == 0 or torch.sum(gt_arr) > THRESHOLD

    def get_image_data(self):
        if self.train_split_percent != 1:
            path_to_splits_file = os.path.join(self.source, SPLITS_FILENAME)
            if not os.path.isfile(path_to_splits_file):
                raise FileNotFoundError(
                    f"A train split percent of {self.train_split_percent} was requested but no \
                    file specifying the dataset split for this split percentage was found. \
                    Please see the README for information on this required file."
                )
            with open(path_to_splits_file) as f:
                splits = json.load(f)
                if str(self.train_split_percent) not in splits.keys():
                    raise KeyError(
                        f"A train split percent of {self.train_split_percent} was requested \
                        but no specification for this split was found in the splits file. Instead, \
                        only the following splits were found: {splits.keys()}."
                    )
                train_split = set(splits[str(self.train_split_percent)]["train"])
                test_split = set(splits[str(self.train_split_percent)]["test"])
                matroid_train_split = set(splits[str(self.train_split_percent)]["matroid_train"])

        datapath = os.path.join(self.source, "data")
        maskpath = os.path.join(self.source, "label")
        # images_to_exclude_path = os.path.join(self.source, "matroid_detector_train_set.json")

        # with open(images_to_exclude_path) as f:
        #     data = json.load(f)
        #     images_to_exclude = [image["fileName"] for image in data["images"]]

        data_to_iterate = []
        for img in os.listdir(datapath):
            if 'KGT' in img:
                continue
            # if img in images_to_exclude:
            #     print("Excluding", img)
            #     continue

            gt = os.path.join(maskpath, f'{os.path.splitext(img)[0]}.json')
            if not os.path.isfile(gt):
                gt = None
            if self.train_split_percent == 1 and \
                ((self.split == DatasetSplit.TRAIN and gt is None) or \
                (self.split == DatasetSplit.TEST and gt is not None)):
                data_to_iterate.extend(
                    [(os.path.join(datapath, img), gt, i) for i in range(self.patches_per_image)]
                )
            elif self.train_split_percent != 1 and \
                ((self.split == DatasetSplit.TRAIN and img in train_split) or \
                    (self.split == DatasetSplit.TEST and img in test_split) or \
                    (self.split == DatasetSplit.MATROID_TRAIN and img in matroid_train_split)):
                data_to_iterate.extend(
                    [(os.path.join(datapath, img), gt, i) for i in range(self.patches_per_image)]
                )

        assert len(data_to_iterate) > 0, "data_to_iterate was empty while loading the dataset"

        return data_to_iterate
