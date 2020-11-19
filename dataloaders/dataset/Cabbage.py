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


class CabbageDataset(Dataset):
    """
    Cabbage dataset
    """
    CAB_CLASSES = ['cabbage', 'not_cabbage']
    NUM_CLASSES = 2

    def __init__(self, args, base_dir=Path.db_root_dir('cabbage'), split='train'):
        """
        :param base_dir: path to cabbage dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.args = args
        self._base_dir = base_dir
        self.split = split

        if self.split == 'train':
            self.train_dir = os.path.join(self._base_dir, 'train')
            self.train_folder_list = sorted(os.listdir(self.train_dir))
            self.train_data_path = [
                '/'.join([self.train_dir, self.train_folder_list[i]])
                for i in range(len(self.train_folder_list))]

        elif self.split == 'val':
            self.val_dir = os.path.join(self._base_dir, 'val')
            self.val_folder_list = sorted(os.listdir(self.val_dir))
            self.val_data_path = [
                '/'.join([self.val_dir, self.val_folder_list[i]])
                for i in range(len(self.val_folder_list))]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        _img, _target = self.make_img_gt_point_pair(idx)
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

    def make_img_gt_point_pair(self, idx):
        if self.split == 'train':
            _img = Image.open(os.path.join(self.train_data_path[idx], 'input.png')).convert('RGB')
            _target = Image.open(os.path.join(self.train_data_path[idx], 'output.png')).convert('L')

            return _img, _target

        elif self.split == 'val':
            _img = Image.open(os.path.join(self.val_data_path[idx], 'input.png')).convert('RGB')
            _target = Image.open(os.path.join(self.val_data_path[idx], 'output.png')).convert('L')

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
            transforms.Resize((self.args.base_size, self.args.base_size))])

        _img, _target = resized_transforms(sample['image']), resized_transforms(sample['label'])

        samples = {'image': _img,
                   'label': _target}

        return self.transform_totensor(samples)

    def transform_unet(self, sample):
        resized_transforms = transforms.Compose([
            transforms.Resize((self.args.base_size, self.args.base_size))])

        _img, _target = resized_transforms(sample['image']), resized_transforms(sample['label'])

        _img = self.mirrored_input(_img, e=92)
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

        _img, _target = resized_transforms(sample['image']), resized_transforms(sample['label'])

        samples = {'image': _img,
                   'label': _target}

        return self.transform_totensor(samples)


# confirm Dataload & Data Augmentation
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cabbage_train = CabbageDataset(args, split='train')

    dataloader = DataLoader(cabbage_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample['image'].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cabbage')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
