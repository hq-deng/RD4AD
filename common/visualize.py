import datetime
import glob
import json
import os

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve, roc_auc_score

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x):
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def export_hist(save_path, gts, scores, threshold):
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    Y = scores.flatten()
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.75, histtype='barstacked')
    image_file = os.path.join(save_path, 'hist_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

def export_test_images(save_path, test_img, image_names, gts, scores, threshold, comparison_images):
    image_dirs = os.path.join(save_path, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    # images
    anomaly_image_names = []
    anomaly_image_aurocs = []
    anomaly_image_defect_percents = []
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        scores_norm = 1.0/scores.max()
        for i in range(num):
            print(f"Scores i shape : {scores[i].shape}")
            img, name = test_img[i].squeeze(), image_names[i]
            img = denormalization(img)
            # gts
            gt_mask = gts[i].squeeze() # Make it just (height, width)

            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(2, 2, figsize=(10*cm, 10*cm))
            for row in ax_img:
                for ax_i in row:
                    ax_i.axes.xaxis.set_visible(False)
                    ax_i.axes.yaxis.set_visible(False)
                    ax_i.spines['top'].set_visible(False)
                    ax_i.spines['right'].set_visible(False)
                    ax_i.spines['bottom'].set_visible(False)
                    ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0, 0].imshow(gt_img)
            ax_img[1, 0].imshow(score_map, cmap='jet', norm=norm)
            ax_img[0, 1].imshow(score_img)
            comparison_image_path = os.path.join(
                comparison_images, os.path.splitext(name[0])[0] + ".*"
            )
            files = glob.glob(comparison_image_path)
            if len(files) > 0:
                comparison_image = Image.open(files[0])
                ax_img[1, 1].imshow(comparison_image)
            visualization_name = '{}_{:08d}.png'.format(name, i)
            image_file = os.path.join(image_dirs, visualization_name)
            fig_img.suptitle(scores[i].max())
            fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            if np.sum(gt_mask) > 0:
                auroc = roc_auc_score(gt_mask.flatten(), score_map.flatten())
                anomaly_image_defect_percents.append(np.sum(gt_mask) / gt_mask.size)
                anomaly_image_names.append(visualization_name)
                anomaly_image_aurocs.append(auroc)

    defect_auroc_info_file = os.path.join(save_path, 'defect_auroc_info.json')
    with open(defect_auroc_info_file, 'w') as f:
        data = {
            'names': anomaly_image_names,
            'aurocs': anomaly_image_aurocs,
            'defect_percents': anomaly_image_defect_percents,
        }
        json.dump(data, f)


def visualize(
    gt_label, score_label, gt_mask, super_mask, test_image_list, image_names, save_path, comparison_images
):
    precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    det_threshold = thresholds[np.argmax(f1)]
    print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
    export_test_images(
        save_path, test_image_list, image_names, gt_mask, super_mask, seg_threshold, comparison_images
    )
    export_test_images(
        f"{save_path}_det_threshold", test_image_list, image_names, gt_mask, super_mask, det_threshold, comparison_images
    )
    export_hist(save_path, gt_mask, super_mask, seg_threshold)
