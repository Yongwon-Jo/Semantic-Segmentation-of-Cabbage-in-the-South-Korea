from __future__ import print_function, division

import os
import numpy as np

from PIL import Image
from skimage import io
# from skimage import transform

from mypath import Path
from dataloaders import custom_transforms as tr

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Cabbage5ChannelDataset(Dataset):
    """
    Cabbage 5 Channel dataset
    """
    CAB_CLASSES = ['cabbage', 'not_cabbage']
    NUM_CLASSES = 2

    def __init__(self, args, base_dir=Path.db_root_dir('cabbage5channel'), split='train'):
        """
        :param base_dir: path to cabbage 5 channel dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.args = args
        self._base_dir = base_dir
        self.split = split

        if self.split == 'train':
            self.train_dir = os.path.join(self._base_dir, "train")
            self.train_folder_list = sorted(os.listdir(self.train_dir))
            self.train_data_path = [
                "/".join([self.train_dir, self.train_folder_list[i]])
                for i in range(len(self.train_folder_list))
            ]

        elif self.split == 'val':
            self.val_dir = os.path.join(self._base_dir, "val")
            self.val_folder_list = sorted(os.listdir(self.val_dir))
            self.val_data_path = [
                "/".join([self.val_dir, self.val_folder_list[i]])
                for i in range(len(self.val_folder_list))
            ]

        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        _img, _target = self._make_img_gt_point_pair(idx)
        sample = {'image': _img,
                  'label': _target}

        if self.args.models == 'DeepLab':

            if self.split == 'train':
                return self.transform_dlv(sample)

            elif self.split == 'val':
                return self.transform_dlv(sample)

        elif self.args.models == 'SegNet':

            if self.split == 'train':
                return self.transform_se(sample)

            elif self.split == 'val':
                return self.transform_se(sample)

        elif self.args.models == 'UNet':

            if self.split == 'train':
                return self.transform_unet(sample)

            elif self.split == 'val':
                return self.transform_unet(sample)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_folder_list)

        elif self.split == 'val':
            return len(self.val_folder_list)

    def _make_img_gt_point_pair(self, idx):
        if self.split == 'train':
            _img = np.load(os.path.join(self.train_data_path[idx], "input.npy"))
            _target = Image.open(os.path.join(self.train_data_path[idx], "output.png")).convert("L")

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

        elif self.split == 'val':
            _img = np.load(os.path.join(self.val_data_path[idx], "input.npy"))
            _target = Image.open(os.path.join(self.val_data_path[idx], "output.png")).convert("L")

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

        _img = original_transforms(sample['image'])
        _target = target_transforms(sample['label'])

        return {'image': _img,
                'label': _target}

    def transform_se(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))])

        _img, _target = sample['image'], resized_transforms(sample['label'])

        samples = {'image': _img,
                   'label': _target}

        return self.transform_totensor(samples)

    def transform_unet(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))])

        _img, _target = self.mirrored_input(sample['image'], e=92), resized_transforms(sample['label'])

        samples = {'image': _img,
                   'label': _target}

        return self.transform_totensor(samples)

    def mirrored_input(self, x, e=92):
        '''input size: 572 -> output size: 388 '''
        # w, h = x.shape
        w, h = np.shape(x)[0], np.shape(x)[1]

        y = np.zeros((h + e * 2, w + e * 2, np.shape(x)[2]))
        y[e: h + e, e: w + e] = x
        y[e: e + h, 0: e] = np.flip(y[e: e + h, e: 2 * e], 1)  # flip vertically
        y[e: e + h, e + w: 2 * e + w] = np.flip(y[e: e + h, w: e + w], 1)
        y[0: e, 0: 2 * e + w] = np.flip(y[e: 2 * e, 0: 2 * e + w], 0)  # flip horizontally
        y[e + h: 2 * e + h, 0: 2 * e + w] = np.flip(y[h: e + h, 0: 2 * e + w], 0)

        return y

    def transform_dlv(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))])

        _img, _target = sample['image'], resized_transforms(sample['label'])

        samples = {'image': _img,
                   'label': _target}

        return self.transform_totensor(samples)
