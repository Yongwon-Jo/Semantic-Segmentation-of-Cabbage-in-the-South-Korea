import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from mypath import Path
from dataloaders import make_data_loader
from torch.utils.data import DataLoader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.segnet import *

from utils.args import segnet_argparser
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
# from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import warnings
warnings.filterwarnings(action='ignore')


class TestCabbageDataset(Dataset):
    def __init__(self, args, path, phase="test"):
        self.args = args
        self.phase = phase
        self.path = path

        self.test_folders = sorted(os.listdir(path))
        self.test_files = []
        for folder in self.test_folders:
            self.test_files.append(os.path.join(self.path, folder))

    def __getitem__(self, idx):
        _img, _target = self.make_img_gt_point_pair(idx)
        sample = {"image": _img,
                "label": _target}
        
        return self.transform_se(sample)

    def __len__(self):
        return len(self.test_files)

    def make_img_gt_point_pair(self, idx):
        _img = Image.open(os.path.join(self.test_files[idx], "input.png")).convert("RGB")
        _target = Image.open(os.path.join(self.test_files[idx], "output.png")).convert("L")

        return _img, _target
    
    def transform_totensor(self, sample):
        original_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.221, 0.413, 0.177],
                                 std=[0.179, 0.160, 0.135])])

        target_transforms = transforms.Compose([
            transforms.ToTensor()])

        _img, _target = original_transforms(sample['image']), target_transforms(sample['label'])

        return {'image': _img,
                'label': _target} 

    def transform_se(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))
        ])

        _img, _target = resized_transforms(sample["image"]), resized_transforms(sample["label"])

        sample = {"image": _img,
                "label": _target}
        
        return self.transform_totensor(sample)


class TestCabbageDataset5channel(Dataset):
    def __init__(self, args, path, phase="test"):
        self.args = args
        self.phase = phase
        self.path = path

        self.test_folders = sorted(os.listdir(path))
        self.test_files = []
        for folder in self.test_folders:
            self.test_files.append(os.path.join(self.path, folder))

    def __getitem__(self, idx):
        _img, _target = self.make_img_gt_point_pair(idx)
        sample = {"image": _img,
                "label": _target}
        
        return self.transform_segnet(sample)

    def __len__(self):
        return len(self.test_files)

    def make_img_gt_point_pair(self, idx):
        _img = np.load(os.path.join(self.test_files[idx], "input.npy"))
        _target = Image.open(os.path.join(self.test_files[idx], "output.png")).convert("L")

        for i in range(_img.shape[2]):
            if i == 0:
                tmp = _img[:, :, i]
                tmp = Image.fromarray(tmp).resize((self.args.base_size, self.args.base_size))
                blank = np.array(tmp)
            
            elif i != 0:
                tmp = _img[:, :, i]
                tmp = Image.fromarray(tmp).resize((self.args.base_size, self.args.base_size))

                blank = np.dstack((blank, np.array(tmp)))

        return blank, _target
    
    def transform_totensor(self, sample):
        original_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.039, 0.083, 0.054, 0.632, 0.180],
                                 std=[0.024, 0.034, 0.043, 0.183, 0.057])])

        target_transforms = transforms.Compose([
            transforms.ToTensor()])

        _img, _target = original_transforms(sample['image']), target_transforms(sample['label'])
        _img = torch.tensor(_img, dtype=torch.float32)
        return {'image': _img,
                'label': _target} 

    def transform_segnet(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))
        ])

        _img, _target = sample["image"], resized_transforms(sample["label"])
        sample = {"image": _img,
                "label": _target}
        
        return self.transform_totensor(sample)



def IoU(pred, mask):
    area_pred = np.where(pred == 1)[0]
    area_mask = np.where(mask == 1)[0]

    return len(np.intersect1d(area_pred, area_mask)) / len(np.union1d(area_pred, area_mask))


def inference(exp_num):
    parser = segnet_argparser()
    args = parser.parse_args([])
    args.cuda = torch.device("cuda:0")
    
    args.input_channel = 5

    model = SegNet(args, num_classes=2)
    model.to(args.cuda)

    results = torch.load(os.path.join(Path.db_root_dir("cabbage5channel"), 
                                'segnet-normal',
                                exp_num,
                                'val_checkpoint.pth'),
                                map_location="cuda:0")

    model.load_state_dict(results["val_state_dict"])
    model.eval()

    val_path = os.path.join(Path.db_root_dir("cabbage5channel"), "val")
    VAL_save = os.path.join(Path.db_root_dir("cabbage5channel"), 
                                'segnet-normal',
                                exp_num,
                                'test_MBS_val')
    if not os.path.exists(VAL_save):
        os.mkdir(VAL_save)

    MBS_path = os.path.join(Path.db_root_dir("cabbage5channel"), "MBS")
    MBS_save = os.path.join(Path.db_root_dir("cabbage5channel"), 
                                'segnet-normal',
                                exp_num,
                                'test_MBS')
    if not os.path.exists(MBS_save):
        os.mkdir(MBS_save)

    GNM_path = os.path.join(Path.db_root_dir("cabbage5channel"), "GNM")
    GNM_save = os.path.join(Path.db_root_dir("cabbage5channel"), 
                                'segnet-normal',
                                exp_num,
                                'test_GNM')
    if not os.path.exists(GNM_save):
        os.mkdir(GNM_save)

    ABD_path = os.path.join(Path.db_root_dir("cabbage5channel"), "ABD")
    ABD_save = os.path.join(Path.db_root_dir("cabbage5channel"), 
                                'segnet-normal',
                                exp_num,
                                'test_ABD')
    if not os.path.exists(ABD_save):
        os.mkdir(ABD_save)

    val_dataset = TestCabbageDataset5channel(args, val_path, phase='test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for i, sample in tqdm(enumerate(val_loader)):
        img = sample['image'].to(device=args.cuda)
        target = sample['label'].to(device=args.cuda)

        with torch.no_grad():
            output = model(img)
        
        target = target.cpu().numpy().squeeze()
        target_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(target==1)

        target_segmap[idx] = 0, 175, 0
        target_segmap = np.uint8(target_segmap)

        _, pred = torch.max(output, dim=1)
        pred = pred.cpu().numpy().squeeze()
        pred_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(pred==1)

        pred_segmap[idx] = 0, 175, 0
        pred_segmap = np.uint8(pred_segmap)

        if not os.path.exists(os.path.join(VAL_save, 'mbs_gt')):
            os.mkdir(os.path.join(VAL_save, 'mbs_gt'))
        if not os.path.exists(os.path.join(VAL_save, 'mbs_pred')):
            os.mkdir(os.path.join(VAL_save, 'mbs_pred'))

        plt.imsave(os.path.join(VAL_save, 'mbs_gt', f"mbs_gt_{str(i)}.png"), target_segmap)
        
        iou = round(IoU(pred.reshape(-1), target.reshape(-1)), 3)
        plt.imsave(os.path.join(VAL_save, 'mbs_pred', f"mbs_pred_{i}_{iou}.png"), pred_segmap)



    mbs_dataset = TestCabbageDataset5channel(args, MBS_path, phase='test')
    mbs_loader = DataLoader(mbs_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for i, sample in tqdm(enumerate(mbs_loader)):
        if i % 6 == 0:
            img = sample['image'].to(device=args.cuda)
            target = sample['label'].to(device=args.cuda)

            with torch.no_grad():
                output = model(img)
            
            target = target.cpu().numpy().squeeze()
            target_segmap = np.zeros((args.base_size, args.base_size, 3))
            idx = np.where(target==1)

            target_segmap[idx] = 0, 175, 0
            target_segmap = np.uint8(target_segmap)

            _, pred = torch.max(output, dim=1)
            pred = pred.cpu().numpy().squeeze()
            pred_segmap = np.zeros((args.base_size, args.base_size, 3))
            idx = np.where(pred==1)

            pred_segmap[idx] = 0, 175, 0
            pred_segmap = np.uint8(pred_segmap)

            if not os.path.exists(os.path.join(MBS_save, 'mbs_gt')):
                os.mkdir(os.path.join(MBS_save, 'mbs_gt'))
            if not os.path.exists(os.path.join(MBS_save, 'mbs_pred')):
                os.mkdir(os.path.join(MBS_save, 'mbs_pred'))

            plt.imsave(os.path.join(MBS_save, 'mbs_gt', f"mbs_gt_{str(i)}.png"), target_segmap)
            
            iou = round(IoU(pred.reshape(-1), target.reshape(-1)), 3)
            plt.imsave(os.path.join(MBS_save, 'mbs_pred', f"mbs_pred_{i}_{iou}.png"), pred_segmap)



    gnm_dataset = TestCabbageDataset5channel(args, GNM_path, phase='test')
    gnm_loader = DataLoader(gnm_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for i, sample in tqdm(enumerate(gnm_loader)):
        img = sample['image'].to(device=args.cuda)
        target = sample['label'].to(device=args.cuda)

        with torch.no_grad():
            output = model(img)
        
        target = target.cpu().numpy().squeeze()
        target_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(target==1)

        target_segmap[idx] = 0, 175, 0
        target_segmap = np.uint8(target_segmap)

        _, pred = torch.max(output, dim=1)
        pred = pred.cpu().numpy().squeeze()
        pred_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(pred==1)

        pred_segmap[idx] = 0, 175, 0
        pred_segmap = np.uint8(pred_segmap)

        if not os.path.exists(os.path.join(GNM_save, 'gnm_gt')):
            os.mkdir(os.path.join(GNM_save, 'gnm_gt'))
        if not os.path.exists(os.path.join(GNM_save, 'gnm_pred')):
            os.mkdir(os.path.join(GNM_save, 'gnm_pred'))

        plt.imsave(os.path.join(GNM_save, 'gnm_gt', f"gnm_gt_{str(i)}.png"), target_segmap)
        
        iou = round(IoU(pred.reshape(-1), target.reshape(-1)), 3)
        plt.imsave(os.path.join(GNM_save, 'gnm_pred', f"gnm_pred_{i}_{iou}.png"), pred_segmap)


    abd_dataset = TestCabbageDataset5channel(args, ABD_path, phase='test')
    abd_loader = DataLoader(abd_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for i, sample in tqdm(enumerate(abd_loader)):
        img = sample['image'].to(device=args.cuda)
        target = sample['label'].to(device=args.cuda)

        with torch.no_grad():
            output = model(img)
        
        target = target.cpu().numpy().squeeze()
        target_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(target==1)

        target_segmap[idx] = 0, 175, 0
        target_segmap = np.uint8(target_segmap)

        _, pred = torch.max(output, dim=1)
        pred = pred.cpu().numpy().squeeze()
        pred_segmap = np.zeros((args.base_size, args.base_size, 3))
        idx = np.where(pred==1)

        pred_segmap[idx] = 0, 175, 0
        pred_segmap = np.uint8(pred_segmap)

        if not os.path.exists(os.path.join(ABD_save, 'abd_gt')):
            os.mkdir(os.path.join(ABD_save, 'abd_gt'))
        if not os.path.exists(os.path.join(ABD_save, 'abd_pred')):
            os.mkdir(os.path.join(ABD_save, 'abd_pred'))

        plt.imsave(os.path.join(ABD_save, 'abd_gt', f"abd_gt_{str(i)}.png"), target_segmap)
        
        iou = round(IoU(pred.reshape(-1), target.reshape(-1)), 3)
        plt.imsave(os.path.join(ABD_save, 'abd_pred', f"abd_pred_{i}_{iou}.png"), pred_segmap)        


for num in range(0, 10):
    exp_num = "experiment_" + str(num)
    inference(exp_num)

