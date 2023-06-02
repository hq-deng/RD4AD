import os.path
from statistics import mean

from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import auc, roc_auc_score
from skimage import measure

# From https://github.com/YoungGod/DFR/blob/master/DFR-source/utils.py
def _rescale(x):
    return (x - x.min()) / (x.max() - x.min())

# Adapted from https://github.com/hq-deng/RD4AD/blob/main/test.py with
# minimal modifications. Notice that this fixes a few bugs in
# https://github.com/YoungGod/DFR/blob/master/DFR-source/utils.py
def calculate_aupro(masks: ndarray, amaps: ndarray, max_fpr=0.3, num_th: int = 300) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    regions_stack = []
    for mask in masks:
        regions_stack.append(measure.regionprops(measure.label(mask)))

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, regions in zip(binary_amaps, regions_stack):
            for region in regions:
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                assert(tp_pixels <= region.area)
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat(
            [
                df,
                pd.DataFrame.from_dict({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]}),
            ],
            ignore_index=True,
        )

    # Normalize FPR and PRO
    df = df[df["fpr"] <= max_fpr]
    df["fpr"] = _rescale(df["fpr"])
    # We only want to rescale fpr, not pro, because in the ideal case TPR is 1 even after removing
    # all cases where FPR is > threshold. To see why this is important, consider the case where our
    # data is TPR=[0, 1, 1] and FPR=[0, 0.3, 1]. We would want the AUPRO in this case to be 0.5. If
    # we were to rescale PRO max, we would also have an AUPRO of 0.5 in the case where TPR=[0, 0.1, 1]
    # and FPR=[0, 0.3, 1], even though in this case the AUPRO should reflect the fact that the model
    # performed way worse.
    # df["pro"] = _rescale(df["pro"])

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def plot_auroc_by_defect_size(masks, amaps, save_dir, img_class, rescale):
    defect_counts = []
    roc_scores = []
    masks = np.array(masks)
    amaps = np.array(amaps)
    for mask, amap in zip(masks, amaps):
        if np.sum(mask) == 0: # Only consider those masks that contain defects
            continue
        auroc = roc_auc_score(mask.flatten(), amap.flatten())
        roc_scores.append(auroc)
        defect_counts.append(np.sum(mask) / rescale)
    plt.figure()
    plt.scatter(defect_counts, roc_scores)
    plt.title(img_class)
    plt.xlabel("# of defective pixels")
    plt.ylabel("AUROC")
    plt.savefig(os.path.join(save_dir, f"{img_class}_plot"))
