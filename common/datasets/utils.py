from enum import Enum

import numpy as np
import torchvision.transforms.functional as F

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    MATROID_TRAIN = "matroid_train"

# From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')
