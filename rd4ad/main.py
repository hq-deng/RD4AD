import datetime
import logging
import time

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import os.path
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import compare_to_matroid, eval_fps, evaluation, patches_compare_to_matroid, patches_evaluation
from torch.nn import functional as F

from common.datasets.bsd_dataset import BSDDataset
from common.datasets.custom_dataset import CustomDataset
from common.datasets.ksdd_dataset import KSDDDataset
from common.datasets.utils import DatasetSplit
from common.profiler import Profiler
from common.utils import Score_Observer, compute_and_store_results
from config import get_args

LOGGER = logging.getLogger(__name__)

dataset_paths = {
    "mvtec": "./datasets/mvtec/",
    "mvtec_realigned": "../datasets/mvtec_realigned",
    "connectors": "../datasets/connectors/processed",
    "automotive": "../datasets/automotive/processed",
    "bsd": "../datasets/BSData",
    "ksdd2": "../datasets/KSDD2"
}

CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = "./results"

def get_run_save_filename(_class_, c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_save_filename = \
        f"wres50_{c.dataset}_{_class_}_lr{c.lr}_bs{c.batch_size}_inp{c.input_size}_train_p{c.train_split_percent}_epochs{c.epochs}_{run_date}"

    return run_save_filename

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape, b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def test(_class_, c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(c.class_names)
    image_size = c.input_size

    ckp_path = c.checkpoint

    run_save_filename = os.path.splitext(os.path.split(ckp_path)[1])[0]

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    dataset_path = dataset_paths[c.dataset]
    if c.dataset == 'mvtec' or c.dataset == 'mvtec_realigned':
        if c.multimodal:
            test_data = []
            for individual_class in _class_:
                test_path = os.path.join(dataset_path, individual_class)
                test_data.append(MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test"))
            test_data = torch.utils.data.ConcatDataset(test_data)
        else:
            test_path = os.path.join(dataset_path, _class_)
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    elif c.dataset == 'connectors' or c.dataset == 'automotive':
        patches_per_row = c.patches_per_row or (5 if c.dataset == 'connectors' else 6)
        patches_per_column = c.patches_per_column or 1
        test_data = CustomDataset(
            dataset_path,
            _class_,
            image_size,
            split=DatasetSplit.TEST, 
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
        )
    elif c.dataset == 'bsd':
        patches_per_row = c.patches_per_row or 1
        patches_per_column = c.patches_per_column or 1
        test_data = BSDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TEST,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
    elif c.dataset == 'ksdd2':
        patches_per_row = c.patches_per_row or 1
        patches_per_column = c.patches_per_column or 1
        test_data = KSDDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TEST,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    print("Test data length", len(test_data))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    save_images_filename = run_save_filename if c.save_segmentation_images else None
    if c.action_type == 'norm-test-fps':
        eval_fps(
            encoder, bn, decoder, test_dataloader, device
        )
    elif c.action_type== 'norm-compare-to-matroid':
        compare_to_matroid(
            c,
            encoder, 
            bn,
            decoder,
            test_dataloader, 
            device,
        )
    elif c.action_type== 'norm-patches-compare-to-matroid':
        patches_compare_to_matroid(
            c,
            encoder, 
            bn,
            decoder,
            test_dataloader, 
            device,
        )
    elif c.action_type == 'norm-patches-test':
        auroc_px, auroc_sp, aupro_px = patches_evaluation(
            c,
            encoder, 
            bn,
            decoder,
            test_dataloader, 
            device,
            _class_,
            save_auroc_filename=run_save_filename, 
            save_images_filename=save_images_filename, 
        )
    else:
        auroc_px, auroc_sp, aupro_px = evaluation(
            c,
            encoder, 
            bn,
            decoder,
            test_dataloader, 
            device,
            _class_,
            save_auroc_filename=run_save_filename, 
            save_images_filename=save_images_filename, 
            save_preds_filename=save_images_filename
        )

        print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)

def train(_class_, c):
    Profiler.restart_profilers(LOGGER)

    print(_class_)
    epochs = c.epochs
    learning_rate = c.lr
    batch_size = c.batch_size
    image_size = c.input_size

    assert not c.load_preds, "Cannot load predictions when training"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_save_filename = get_run_save_filename(_class_, c)
    ckp_path = os.path.join(CHECKPOINT_DIR, f"{run_save_filename}.pth")
    
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    dataset_path = dataset_paths[c.dataset]
    if c.dataset == 'mvtec' or c.dataset == 'mvtec_realigned':
        if c.multimodal:
            train_data = []
            test_data = []
            for individual_class in _class_:
                train_path = os.path.join(dataset_path, individual_class, 'train')
                test_path = os.path.join(dataset_path, individual_class)
                train_data.append(ImageFolder(root=train_path, transform=data_transform))
                test_data.append(MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test"))
            train_data = torch.utils.data.ConcatDataset(train_data)
            test_data = torch.utils.data.ConcatDataset(test_data)
        else:
            print(dataset_path)
            print(_class_)
            train_path = os.path.join(dataset_path, _class_, 'train')
            test_path = os.path.join(dataset_path, _class_)
            train_data = ImageFolder(root=train_path, transform=data_transform)
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    elif c.dataset == 'connectors' or c.dataset == 'automotive':
        patches_per_row = c.patches_per_row or (5 if c.dataset == 'connectors' else 6)
        patches_per_column = c.patches_per_column or 1
        train_data = CustomDataset(
            dataset_path,
            _class_,
            image_size,
            split=DatasetSplit.TRAIN, 
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column,
            train_split_percent=c.train_split_percent,
        )
        test_data = CustomDataset(
            dataset_path, 
            _class_, 
            image_size, 
            split=DatasetSplit.TEST, 
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column,
            train_split_percent=c.train_split_percent,
        )
    elif c.dataset == 'bsd':
        patches_per_row = c.patches_per_row or 1
        patches_per_column = c.patches_per_column or 1
        train_data = BSDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TRAIN,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
        test_data = BSDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TEST,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
    elif c.dataset == 'ksdd2':
        patches_per_row = c.patches_per_row or 1
        patches_per_column = c.patches_per_column or 1
        train_data = KSDDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TRAIN,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
        test_data = KSDDDataset(
            dataset_path,
            image_size,
            split=DatasetSplit.TEST,
            patches_per_row=patches_per_row, 
            patches_per_column=patches_per_column, 
            train_split_percent=c.train_split_percent,
            crop_width=c.crop_width,
            crop_height=c.crop_height,
        )
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    print("Train data length", len(train_data))
    print("Test data length", len(test_data))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    Profiler.measure("After loading encoder")

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    Profiler.measure("After loading decoder architectures")

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    total_train_time_s = 0
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')

    auroc_px, auroc_sp, aupro_px = 0, 0, 0

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for example in train_dataloader:
            if c.dataset == 'connectors' or c.dataset == 'automotive' or c.dataset == 'bsd' or c.dataset == 'ksdd2':
                img, label = example["image"], example["is_anomaly"]
            else:
                img, label = example
            start = time.time()
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            total_train_time_s += (time.time() - start)
        Profiler.measure("After training for one epoch")
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 1 == 0:
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
            # auroc_px, auroc_sp, aupro_px = evaluation(
            #     c, encoder, bn, decoder, test_dataloader, device, _class_=_class_, save_auroc_filename=run_save_filename,
            # )
            # Profiler.measure("After predicting")
            # Profiler.log_stats(LOGGER)
            # _ = det_roc_obs.update(auroc_sp, epoch)
            # _ = seg_roc_obs.update(auroc_px, epoch)
            # _ = seg_pro_obs.update(aupro_px, epoch)
            # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            
        Profiler.restart_profilers(LOGGER)

    results = {
        "instance_auroc": auroc_sp,
        "full_pixel_auroc": auroc_px,
        "full_pixel_aupro": aupro_px,
        "instance_auroc_best": det_roc_obs.max_score,
        "full_pixel_auroc_best": seg_roc_obs.max_score,
        "full_pixel_aupro_best": seg_pro_obs.max_score,
        "instance_auroc_best_epoch": det_roc_obs.max_epoch,
        "full_pixel_auroc_best_epoch": seg_roc_obs.max_epoch,
        "full_pixel_aupro_best_epoch": seg_pro_obs.max_epoch,
        "train_time_s": total_train_time_s,
        "max_allocated_gpu_memory": Profiler.max_allocated_gpu_memory[0],
        "max_cpu_utilization": Profiler.max_cpu_utilization[0],
        "max_used_ram_percent": Profiler.max_used_ram_percent[0],
        "max_used_ram_bytes": Profiler.max_used_ram_bytes[0],
    }

    compute_and_store_results(
        results.values(), results.keys(), os.path.join(RESULTS_DIR, run_save_filename)
    )

    return results


if __name__ == '__main__':
    setup_seed(111)
    c = get_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if c.multimodal:
        if c.dataset != 'mvtec':
            raise NotImplementedError("Multimodal training is not implemented for non-mvtec datasets")
        else:
            c.class_names = [
                ['zipper', 'hazelnut', 'toothbrush'],
                ['carpet', 'leather', 'wood'],
                ['transistor', 'grid', 'tile'],
                ['metal_nut', 'capsule', 'bottle'],
                ['screw', 'pill', 'cable'],
            ]
    if (
        c.action_type == "norm-test" or
        c.action_type == "norm-patches-test" or
        c.action_type == 'norm-compare-to-matroid' or
        c.action_type == 'norm-patches-compare-to-matroid' or
        c.action_type == 'norm-test-fps'
    ):
        for _class_ in c.class_names:
            test(_class_, c)
    elif c.action_type == "norm-train":
        results = {}
        for i in c.class_names:
            LOGGER.info(f"Training on {i} in dataset {c.dataset}")
            class_results = train(i, c)
            for column in class_results:
                if column not in results:
                    results[column] = [class_results[column]]
                else:
                    results[column].append(class_results[column])
        for column in results:
            results[column] = np.mean(results[column])
        run_save_filename = get_run_save_filename('means', c)
        compute_and_store_results(
            results.values(), results.keys(), os.path.join(RESULTS_DIR, run_save_filename)
        )
    else:
        raise NotImplementedError(
            f"Attempted to run RD4AD with unknown action type {c.action_type}"
        )
            

