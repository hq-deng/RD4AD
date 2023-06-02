import json
import os

import numpy as np
import PIL
import torch
from torchvision import transforms

from common.datasets.utils import DatasetSplit


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SPLITS_FILENAME = "splits.json"
COLORS_FILEPATH = """/home/ubuntu/modelplayground/anomaly-detection/datasets/automotive/class_color.json"""


class CustomDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Custom.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        split=DatasetSplit.TRAIN,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Custom data folder.
            classname: [str or None]. Name of Custom class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. custom_dataset.DatasetSplit.TRAIN. Note that
                   custom_dataset.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        if isinstance(classname, list):
            self.classnames_to_use = classname
        else:
            self.classnames_to_use = [classname]

        self.patches_per_image = kwargs.get('patches_per_image', None)
        self.patches_per_row = kwargs.get('patches_per_row', self.patches_per_image)
        assert(self.patches_per_row is not None)
        self.patches_per_column = kwargs.get('patches_per_column', 1)
        if self.patches_per_image is None:
            self.patches_per_image = self.patches_per_row * self.patches_per_column

        self.train_split_percent = kwargs.get('train_split_percent', 1)

        self.transform_img = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            # Open the mask as black and white, since there may be different colors
            # assigned to different semantic classes of defects.
            transforms.Lambda(lambda x : x.convert('L')),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : (x != 0).to(dtype=x.dtype)),
        ]
        # For datasets annotated by annotation partner, there may be some annotations marked
        # as hard. This gives us the option to ignore these annotations.
        # By default, these harder annotations are ignored.
        if not kwargs.get('retain_hard', False):
            self.transform_mask.insert(1, transforms.Lambda(self._remove_hard))

        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # Since anomaly detection models tend to perform best on larger defects, this allows
        # us to remove examples where the total number of defective pixels is very small.
        if kwargs.get('filter_big_defects', False):
            self.data_to_iterate = list(filter(self._filter_big_defects, self.data_to_iterate))

    def __getitem__(self, idx):
        classname, _, image_path, mask_path, index = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

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
            "classname": classname,
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
        gt = img_tuple[3]
        index = img_tuple[4]
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

    def _remove_hard(self, gt_img):
        with open(COLORS_FILEPATH) as f:
            colors = json.load(f)
            gt_arr = np.asarray(gt_img)

            new_gt_arr = gt_arr.copy()
            for defect_kind, pixel_value in colors.items():
                if defect_kind.endswith('hard'):
                    new_gt_arr[np.all(new_gt_arr == pixel_value, axis=-1)] = (0, 0, 0)

            return PIL.Image.fromarray(new_gt_arr).convert('L')

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            if self.train_split_percent != 1:
                path_to_splits_file = os.path.join(self.source, classname, SPLITS_FILENAME)
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

            classpath = os.path.join(self.source, classname, "imgs")
            maskpath = os.path.join(self.source, classname, "gt")

            imgpaths_per_class[classname] = []
            maskpaths_per_class[classname] = []

            for img in os.listdir(classpath):
                gt = os.path.join(maskpath, f'{os.path.splitext(img)[0]}.png')
                if not os.path.isfile(gt):
                    gt = None
                if self.train_split_percent == 1 and \
                   ((self.split == DatasetSplit.TRAIN and 'Good' in img) or \
                    (self.split == DatasetSplit.TEST and 'Good' not in img)):
                    imgpaths_per_class[classname].append({
                        'img': os.path.join(classpath, img),
                        'gt': gt
                    })
                elif self.train_split_percent != 1 and \
                    ((self.split == DatasetSplit.TRAIN and img in train_split) or \
                     (self.split == DatasetSplit.TEST and img not in train_split)):
                    imgpaths_per_class[classname].append({
                        'img': os.path.join(classpath, img),
                        'gt': gt
                    })

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for img_dict in imgpaths_per_class[classname]:
                data_to_iterate.extend(
                    [
                        [
                            classname,
                            # PatchCore expects there to be something here,
                            # but we do not use this for this dataset.
                            None, 
                            img_dict['img'],
                            img_dict['gt'],
                            i
                        ] for i in range(self.patches_per_image)
                    ]
                )

        assert len(data_to_iterate) > 0, "data_to_iterate was empty while loading the dataset"

        return imgpaths_per_class, data_to_iterate
