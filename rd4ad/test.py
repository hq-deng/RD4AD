import json
import os
import pickle
from pprint import pprint
from statistics import mean

import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
from skimage.transform import resize
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
from PIL import Image
from tqdm import tqdm

from anomaly_map import cal_anomaly_map
from common.evaluation.pixelwise import calculate_aupro, plot_auroc_by_defect_size
from common.evaluation.bounding_box import GTKind, BoundingBoxMetrics, PredKind
from common.utils import t2np
from common.visualize import visualize

AUROC_DIR = 'auroc_plots'
SEGMENTATION_DIR = 'segmentation_images'
PREDS_DIR = 'image_preds'

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def eval_fps(encoder, bn, decoder, dataloader, device, dataset=None):
    bn.eval()
    decoder.eval()
    with torch.no_grad():
        # warm-up
        # TODO(@nraghuraman-matroid) Make a minor change to have this work for the custom
        # dataset as well.
        for image, _, _, _ in dataloader:
            # data
            image = image.to(device) # single scale
            _ = decoder(bn(encoder(image)))  # BxCxHxW
        torch.cuda.synchronize()
        starter, ender = (
            torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        )
        time_all = 0
        for example in dataloader:
            if dataset == 'connectors' or dataset == 'automotive':
                img, label, gt, img_name = example["image"], example["is_anomaly"], example["mask"], example["image_name"]
            else:
                img, gt, label, img_name = example
            img = img.to(device)
            starter.record()
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            ender.record()
            torch.cuda.synchronize()
            time_all += starter.elapsed_time(ender)
        torch.cuda.synchronize()
    fps = len(dataloader.dataset) / (time_all / 1000)
    print("FPS", fps)

def evaluation(
    c,
    encoder, 
    bn, 
    decoder, 
    dataloader,
    device,
    _class_=None, 
    save_auroc_filename=None, 
    save_images_filename=None,
    save_preds_filename=None,
):
    print("In evaluation vanilla")
    bn.eval()
    decoder.eval()

    dataset = c.dataset
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gt_list_imgs = []
    pr_list_imgs = []
    aupro_list = []
    image_list = []
    image_names = []

    if save_preds_filename is not None:
        save_preds_path = os.path.join(PREDS_DIR, save_preds_filename)
        os.makedirs(save_preds_path, exist_ok=True)

    with torch.no_grad():
        anomaly_scores_list = []
        anomaly_scores_map = {}
        anomaly_scores_pre_gaussian = []
        anomaly_scores_pre_smoothing = []
        per_m_max = [[] for i in range(3)]
        for example in tqdm(dataloader):
            if dataset == 'connectors' or dataset == 'automotive' or dataset == 'bsd' or dataset == 'ksdd2':
                img, label, gt, img_name = (
                    example["image"], 
                    example["is_anomaly"], 
                    example["mask"], 
                    example["image_name"][0],
                )
            else:
                # img, gt, label, img_name = example
                img, img_name = example
            
            if c.load_preds:
                anomaly_map = np.load(os.path.join(save_preds_path, img_name + ".npy"))
            else:
                img = img.to(device)
                inputs = encoder(img)
                outputs = decoder(bn(inputs))
                anomaly_map, cal_amap_list = cal_anomaly_map(inputs, outputs, img.shape[-2:], amap_mode='a')

                for i, amap in enumerate(cal_amap_list):
                    per_m_max[i].append(amap.max())

                anomaly_scores_pre_smoothing.append(anomaly_map.max())
                
                anomaly_map /= (2 * len(outputs))


                anomaly_scores_pre_gaussian.append(anomaly_map.max())
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                # print(f"DEBUG: Min: {anomaly_map.min()}, Max: {anomaly_map.max()}")
                # print(f"DEBUG: Length outputs: {len(outputs)}")
                anomaly_scores_list.append(anomaly_map.max())
                anomaly_scores_map[img_name] = anomaly_map.max()

                np.save(os.path.join(img_name[0][:-4]+".npy"), anomaly_map)

                if save_preds_filename is not None:
                    np.save(os.path.join(save_preds_path, img_name[0][:-4]), anomaly_map)

            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            # if label.item()!=0:
            #     aupro_list.append(calculate_aupro(gt.squeeze(0).cpu().numpy().round().astype(np.int32),
            #                                       anomaly_map[np.newaxis,:,:]))

            # gt_list_sp.append(np.max(gt.cpu().numpy().round().astype(np.int32)))
            # pr_list_sp.append(np.max(anomaly_map))

            # # To avoid using too much memory, rescale gt and anomaly maps
            # image_width, image_height = anomaly_map.shape[-1], anomaly_map.shape[-2]
            # if image_width * image_height * len(dataloader) > 200000000:
            #     rescale = 1/16
            #     new_width, new_height = image_width // 4, image_height // 4
            #     gt = torch.tensor(
            #         resize(gt.squeeze(), (new_height, new_width))[np.newaxis, np.newaxis, :, :]
            #     )
            #     anomaly_map = resize(anomaly_map, (new_height, new_width))
            # else:
            #     rescale = 1

            # gt_list_px.extend(gt.cpu().numpy().round().astype(np.int32).ravel())
            # pr_list_px.extend(anomaly_map.ravel())
            # gt_list_imgs.append(gt.cpu().numpy().round().astype(np.int32))
            # pr_list_imgs.append(anomaly_map)

            # if save_images_filename is not None:
            #     img = t2np(img)
            #     if image_width * image_height * len(dataloader) > 200000000:
            #         new_width, new_height = image_width // 4, image_height // 4
            #         img = resize(img, (1, 3, new_height, new_width))
            #     image_list.append(img)
            #     image_names.append(img_name)

        # Dump map pickle
        # print(f"Dumping pickle maps: {len(anomaly_scores_map)}")
        # with open('mot17_04_anomaly_scores.pickle', 'wb') as handle:
        #     pickle.dump(anomaly_scores_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print(f"DEBUG: BEFORE Smoothing Mean : {np.mean(anomaly_scores_pre_smoothing)}, SD : {np.std(anomaly_scores_pre_smoothing)}, Max : {np.max(anomaly_scores_pre_smoothing)}")
        print(f"DEBUG: BEFORE Mean : {np.mean(anomaly_scores_pre_gaussian)}, SD : {np.std(anomaly_scores_pre_gaussian)}, Max : {np.max(anomaly_scores_pre_gaussian)}")
        print(f"DEBUG: AFTER Mean : {np.mean(anomaly_scores_list)}, SD : {np.std(anomaly_scores_list)}, Max : {np.max(anomaly_scores_list)}")
        # for i in range(len(per_m_max[0])):
        #     print(f"DEBUG: Per Amap contribution : {np.mean(per_m_max[0][i]), np.mean(per_m_max[1][i]), np.mean(per_m_max[2][i])}")

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        if len(np.unique(np.array(gt_list_sp))) == 1:
            print("Only one class of images present in test set")
            auroc_sp = np.nan
        else:
            auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    if save_auroc_filename is not None:
        auroc_save_dir = os.path.join(AUROC_DIR, save_auroc_filename)
        os.makedirs(auroc_save_dir, exist_ok=True)
        plot_auroc_by_defect_size(
            gt_list_imgs, pr_list_imgs, auroc_save_dir, _class_, rescale
        )
    if save_images_filename is not None:
        segmentation_save_dir = os.path.join(SEGMENTATION_DIR, save_images_filename)
        os.makedirs(segmentation_save_dir, exist_ok=True)
        visualize(
            gt_list_sp,
            pr_list_sp,
            np.array(gt_list_imgs), 
            np.array(pr_list_imgs), 
            image_list, 
            image_names, 
            segmentation_save_dir, 
            c.comparison_images
        )
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

def patches_evaluation(
    c,
    encoder, 
    bn, 
    decoder, 
    dataloader,
    device,
    _class_=None, 
    save_auroc_filename=None, 
    save_images_filename=None,
    save_preds_filename=None,
):
    bn.eval()
    decoder.eval()

    dataset = c.dataset
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gt_list_imgs = []
    pr_list_imgs = []
    aupro_list = []
    image_list = []
    image_names = []

    if save_preds_filename is not None:
        save_preds_path = os.path.join(PREDS_DIR, save_preds_filename)
        os.makedirs(save_preds_path, exist_ok=True)

    with torch.no_grad():
        # In order to evaluate patch performance, we need to unpatchify the gt, the img patches, and predictions
        # and reconstruct a full image/anomaly map/gt.
        gt_patches_so_far = []
        img_patches_so_far = []
        pred_patches_so_far = []
        for example in tqdm(dataloader):
            if dataset == 'connectors' or dataset == 'automotive' or dataset == 'bsd' or dataset == 'ksdd2':
                img, label, gt, img_name = (
                    example["image"], 
                    example["is_anomaly"], 
                    example["mask"], 
                    example["image_name"][0],
                )
            else:
                img, gt, label, img_name = example
            
            if c.load_preds:
                anomaly_map = np.load(os.path.join(save_preds_path, img_name + ".npy"))
            else:
                img = img.to(device)
                inputs = encoder(img)
                outputs = decoder(bn(inputs))
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-2:], amap_mode='a')
                

                if save_preds_filename is not None:
                    np.save(os.path.join(save_preds_path, img_name), anomaly_map)

            gt_patches_so_far.append(gt)
            img_patches_so_far.append(img)
            pred_patches_so_far.append(torch.tensor(anomaly_map))

            # If there is no patching applied to the image, we would expect
            # is_done to always be True. Otherwise, we must accumulate patches
            # until we have filled up the entire image with patches.
            is_done, gt, img, anomaly_map = dataloader.dataset.unpatch(
                gt_patches_so_far, img_patches_so_far, pred_patches_so_far
            )
            if is_done:
                gt_patches_so_far, img_patches_so_far, pred_patches_so_far = [], [], []
                anomaly_map = anomaly_map.numpy()
            else:
                continue

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(calculate_aupro(gt.squeeze(0).cpu().numpy().round().astype(np.int32),
                                                  anomaly_map[np.newaxis,:,:]))

            gt_list_sp.append(np.max(gt.cpu().numpy().round().astype(np.int32)))
            pr_list_sp.append(np.max(anomaly_map))

            # To avoid using too much memory, rescale gt and anomaly maps
            image_width, image_height = anomaly_map.shape[-1], anomaly_map.shape[-2]
            if image_width * image_height * len(dataloader) > 200000000:
                rescale = 1/16
                new_width, new_height = image_width // 4, image_height // 4
                gt = torch.tensor(
                    resize(gt.squeeze(), (new_height, new_width))[np.newaxis, np.newaxis, :, :]
                )
                anomaly_map = resize(anomaly_map, (new_height, new_width))
            else:
                rescale = 1

            gt_list_px.extend(gt.cpu().numpy().round().astype(np.int32).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_imgs.append(gt.cpu().numpy().round().astype(np.int32))
            pr_list_imgs.append(anomaly_map)

            if save_images_filename is not None:
                img = t2np(img)
                if image_width * image_height * len(dataloader) > 200000000:
                    new_width, new_height = image_width // 4, image_height // 4
                    img = resize(img, (1, 3, new_height, new_width))
                image_list.append(img)
                image_names.append(img_name)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        if len(np.unique(np.array(gt_list_sp))) == 1:
            print("Only one class of images present in test set")
            auroc_sp = np.nan
        else:
            auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    if save_auroc_filename is not None:
        auroc_save_dir = os.path.join(AUROC_DIR, save_auroc_filename)
        os.makedirs(auroc_save_dir, exist_ok=True)
        plot_auroc_by_defect_size(
            gt_list_imgs, pr_list_imgs, auroc_save_dir, _class_, rescale
        )
    if save_images_filename is not None:
        segmentation_save_dir = os.path.join(SEGMENTATION_DIR, save_images_filename)
        os.makedirs(segmentation_save_dir, exist_ok=True)
        visualize(
            gt_list_sp,
            pr_list_sp,
            np.array(gt_list_imgs), 
            np.array(pr_list_imgs), 
            image_list, 
            image_names, 
            segmentation_save_dir, 
            c.comparison_images
        )
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

def compare_to_matroid(
    c,
    encoder, 
    bn, 
    decoder, 
    dataloader,
    device,
    save_preds_filename=None,
):
    """
    Currently only support connectors, automotive, bsd, and ksdd2 datasets (mvtec not supported).

    Assumes that the json contains only those files that the matroid detector was not
    also trained on.
    """
    bn.eval()
    decoder.eval()

    dataset = c.dataset

    # Assumes the images are resized/padded to squares
    rd4ad_score = BoundingBoxMetrics(c.input_size, c.input_size)
    matroid_score = BoundingBoxMetrics(c.input_size, c.input_size)

    rd4ad_score_all_test = BoundingBoxMetrics(c.input_size, c.input_size)

    if save_preds_filename is not None:
        save_preds_path = os.path.join(PREDS_DIR, save_preds_filename)
        os.makedirs(save_preds_path, exist_ok=True)

    if not os.path.isfile(c.comparison_bboxes):
        raise FileNotFoundError("Please provide a valid path to predictions \
                                 from a matroid object detector for comparison")

    total_images = len(dataloader)
    total_defective_images = 0

    with open(c.comparison_bboxes) as f:
        matroid_preds = json.load(f)
        with torch.no_grad():
            for example in tqdm(dataloader):
                if dataset == 'connectors' or dataset == 'automotive' or dataset == 'bsd' or dataset == 'ksdd2':
                    img, label, gt, img_name, img_path = (
                        example["image"], 
                        example["is_anomaly"], 
                        example["mask"], 
                        example["image_name"][0],
                        example["image_path"][0],
                    )
                else:
                    raise NotImplementedError(
                        f"Dataset {dataset} not currently suppored for comparison evaluation."
                    )

                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                if torch.sum(gt) > 0:
                    total_defective_images += 1
                # Uncomment the below two lines to skip all images that do
                # not contain any defects in the ground truth.
                # else:
                #     continue

                matroid_pred = matroid_preds.get(os.path.splitext(img_name)[0] + ".jpg", None)
                if matroid_pred is None:
                    matroid_pred = matroid_preds.get(os.path.splitext(img_name)[0] + ".png", None)
                if matroid_pred is not None:
                    # The matroid object detector and the anomaly detection model may have
                    # slightly different test sets. This allows us to only evaluate on their
                    # intersection.
                    width, height = Image.open(img_path).size
                    matroid_score.update(
                        {
                            'height': height, 
                            'width': width, 
                            'bboxes': [pred['bbox'] for pred in matroid_pred],
                        },
                        np.array(gt),
                        pred_kind=PredKind.BBOX_DICT,
                        gt_kind=GTKind.MASK,
                        confs_pred=[list(pred['labels'].values())[0] for pred in matroid_pred],
                    )
                else:
                    assert torch.sum(gt) > 0, img_name

                if c.load_preds:
                    anomaly_map = np.load(os.path.join(save_preds_path, img_name + ".npy"))
                else:
                    img = img.to(device)
                    inputs = encoder(img)
                    outputs = decoder(bn(inputs))
                    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-2:], amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                    if save_preds_filename is not None:
                        np.save(os.path.join(save_preds_path, img_name), anomaly_map)

                assert c.pred_threshold is not None, "No bounding box threshold provided"

                rd4ad_score_all_test.update(
                    np.array(anomaly_map),
                    np.array(gt),
                    pred_kind=PredKind.HEATMAP,
                    gt_kind=GTKind.MASK,
                    pred_threshold=c.pred_threshold,
                )
                if matroid_pred is not None:
                    rd4ad_score.update(
                        np.array(anomaly_map),
                        np.array(gt),
                        pred_kind=PredKind.HEATMAP,
                        gt_kind=GTKind.MASK,
                        pred_threshold=c.pred_threshold,
                    )

    print("Results on only the INTERSECTION test dataset (i.e., those test images that are shared)")
    print("RD4AD results")
    pprint(rd4ad_score.compute())
    print("Matroid detector results")
    pprint(matroid_score.compute())
    print("Results on the the ENTIRE test dataset (some of these images are in Matroid's train set, \
           so we don't evaluate Matroid results here)")
    print("RD4AD results")
    pprint(rd4ad_score_all_test.compute())

    print("Total images:", total_images)
    print("Total defective images:", total_defective_images)

def patches_compare_to_matroid(
    c,
    encoder, 
    bn, 
    decoder, 
    dataloader,
    device,
    save_preds_filename=None,
):
    """
    Currently only support connectors, automotive, bsd, and ksdd2 datasets (mvtec not supported).

    Assumes that the json contains only those files that the matroid detector was not
    also trained on.

    Unlike compare_to_matroid(), this function does not actually calculate metrics on
    Matroid's predictions. This is because the ground truth used by patch inputs might be cropped
    slightly differently, leading to inaccuracies in computing Matroid's metrics.
    """
    bn.eval()
    decoder.eval()

    dataset = c.dataset

    # Assumes the images are resized/padded to squares
    # Note that this may be completely incorrect for the case of rd4ad as
    # inputs will be patchified and the original image might not actually
    # be a square. However, this should not affect the metric calculations.
    rd4ad_score = BoundingBoxMetrics(c.input_size, c.input_size)
    rd4ad_score_all_test = BoundingBoxMetrics(c.input_size, c.input_size)

    if save_preds_filename is not None:
        save_preds_path = os.path.join(PREDS_DIR, save_preds_filename)
        os.makedirs(save_preds_path, exist_ok=True)

    if not os.path.isfile(c.comparison_bboxes):
        raise FileNotFoundError("Please provide a valid path to predictions \
                                 from a matroid object detector for comparison")

    total_images = len(dataloader)
    total_defective_images = 0

    with open(c.comparison_bboxes) as f:
        matroid_preds = json.load(f)
        # In order to evaluate patch performance, we need to unpatchify the gt, the img patches, and predictions
        # and reconstruct a full image/anomaly map/gt.
        gt_patches_so_far = []
        img_patches_so_far = []
        pred_patches_so_far = []
        with torch.no_grad():
            for example in tqdm(dataloader):
                if dataset == 'connectors' or dataset == 'automotive' or dataset == 'bsd' or dataset == 'ksdd2':
                    img, label, gt, img_name, img_path = (
                        example["image"], 
                        example["is_anomaly"], 
                        example["mask"], 
                        example["image_name"][0],
                        example["image_path"][0],
                    )
                else:
                    raise NotImplementedError(
                        f"Dataset {dataset} not currently suppored for comparison evaluation."
                    )

                if c.load_preds:
                    anomaly_map = np.load(os.path.join(save_preds_path, img_name + ".npy"))
                else:
                    img = img.to(device)
                    inputs = encoder(img)
                    outputs = decoder(bn(inputs))
                    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-2:], amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                    if save_preds_filename is not None:
                        np.save(os.path.join(save_preds_path, img_name), anomaly_map)

                gt_patches_so_far.append(gt)
                img_patches_so_far.append(img)
                pred_patches_so_far.append(torch.tensor(anomaly_map))

                # If there is no patching applied to the image, we would expect
                # is_done to always be True. Otherwise, we must accumulate patches
                # until we have filled up the entire image with patches.
                is_done, gt, img, anomaly_map = dataloader.dataset.unpatch(
                    gt_patches_so_far, img_patches_so_far, pred_patches_so_far
                )
                if is_done:
                    gt_patches_so_far, img_patches_so_far, pred_patches_so_far = [], [], []
                    anomaly_map = anomaly_map.numpy()
                else:
                    continue

                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                if torch.sum(gt) > 0:
                    total_defective_images += 1
                # Uncomment the below two lines to skip all images that do
                # not contain any defects in the ground truth.
                # else:
                #     continue

                assert c.pred_threshold is not None, "No bounding box threshold provided"

                rd4ad_score_all_test.update(
                    np.array(anomaly_map),
                    np.array(gt),
                    pred_kind=PredKind.HEATMAP,
                    gt_kind=GTKind.MASK,
                    pred_threshold=c.pred_threshold,
                )
                matroid_pred = matroid_preds.get(os.path.splitext(img_name)[0] + ".jpg", None)
                if matroid_pred is None:
                    matroid_pred = matroid_preds.get(os.path.splitext(img_name)[0] + ".png", None)
                if matroid_pred is not None:
                    rd4ad_score.update(
                        np.array(anomaly_map),
                        np.array(gt),
                        pred_kind=PredKind.HEATMAP,
                        gt_kind=GTKind.MASK,
                        pred_threshold=c.pred_threshold,
                    )
                else:
                    assert torch.sum(gt) > 0, img_name

    print("Results on only the INTERSECTION test dataset (i.e., those test images that are shared)")
    print("RD4AD results")
    pprint(rd4ad_score.compute())
    print("Results on the the ENTIRE test dataset (some of these images are in Matroid's train set")
    print("RD4AD results")
    pprint(rd4ad_score_all_test.compute())

    print("Total images:", total_images)
    print("Total defective images:", total_defective_images)
