import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from mypath import Path
from dataloaders import make_data_loader
from modeling.segnet import *

from utils.args import segnet_argparser
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from utils.saver import Saver
from utils.metrics import Evaluator

import warnings
warnings.filterwarnings(action='ignore')


def IoU(pred, mask):
    area_pred = np.where(pred == 1)[0]
    area_mask = np.where(mask == 1)[0]

    return len(np.intersect1d(area_pred, area_mask)) / len(np.union1d(area_pred, area_mask))


class SegNetTrainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        kwargs = {'pin_memory': True}
        self.train_loader, self.val_loader, self.n_class = make_data_loader(args, **kwargs)

        # Define network
        model = SegNet(args, num_classes=self.n_class)

        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Define Criterion
        # Whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset),
                                                args.dataset + '_classes_weights.npy')

            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)

            else:
                weight = calculate_weights_labels(args.dataset, self.train_loader, self.n_class)

            weight = torch.from_numpy(weight.astype(np.float32))

        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Using cuda
        self.model = self.model.to(device=args.cuda)

        # Resuming checkpoint
        self.best_pred = 0.0

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f'=> no checkpoint found at {args.resume}')

            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])

            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        
        print('\n => Epochs %i, learning rate = %.4f, previous best = %.4f' %
              (epoch, self.args.lr, self.best_pred))

        train_losses = []
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.to(device=self.args.cuda), target.to(device=self.args.cuda)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target.squeeze())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        train_loss /= (i + 1)
        train_losses.append(train_loss)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        return train_losses

    def validation(self, epoch):
        self.model.eval()
        val_loss = 0.0

        if not os.path.exists(os.path.join(self.saver.experiment_dir, 'val_gt')):
            os.mkdir(os.path.join(self.saver.experiment_dir, 'val_gt'))

        if not os.path.exists(os.path.join(self.saver.experiment_dir, f'val_pred{epoch + 1}')):
            os.mkdir(os.path.join(self.saver.experiment_dir, f'val_pred{epoch + 1}'))

        val_losses, total_iou = [], []
        with tqdm(total=len(self.val_loader)) as tbar:
            for i, sample in enumerate(self.val_loader):
                image = sample['image'].to(device=self.args.cuda)
                target = sample['label'].to(device=self.args.cuda)

                with torch.no_grad():
                    output = self.model(image)

                loss = self.criterion(output, target.squeeze(0))
                val_loss += loss.item()
                tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
                tbar.update(1)

                target = target.cpu().numpy()
                target = target.squeeze()
                _, predict = torch.max(output, dim=1)
                pred = predict.cpu().numpy()
                pred = pred.squeeze()

                if epoch == 0:
                    plt.imsave(os.path.join(self.saver.experiment_dir, 'val_gt', f'val_gt_{i}.png'), target)

                else:
                    pass

                try:
                    iou = IoU(pred.reshape(-1), target.reshape(-1))  # flatten 1d
                    total_iou.append(iou)

                    plt.imsave(os.path.join(
                        self.saver.experiment_dir, f'val_pred{epoch + 1}',
                        f'val_pred_{i}_{round(iou, 3)}.png'), pred)

                except ZeroDivisionError:
                    print("ZeroDivision")

        total_mean_iou = np.mean(total_iou)
        val_loss /= (i + 1)
        val_losses.append(val_loss)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i + 1))
        print(f'mIoU: {total_mean_iou}')
        print('Loss: %.3f' % val_loss)

        new_pred = total_mean_iou

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'val_state_dict': self.model.state_dict(),
                'val_optimizer': self.optimizer.state_dict(),
                'val_best_pred': self.best_pred,
            }, is_best, filename='val_checkpoint.pth')

        return val_losses, self.saver.experiment_dir

    def rgbsave(self, epoch):
        if epoch == 0 and self.args.dataset == 'cabbage':

            rgb_transforms = transforms.Compose([
                transforms.Resize((self.args.base_size, self.args.base_size)),
                transforms.ToTensor()])

            if not os.path.exists(os.path.join(self.saver.experiment_dir, 'val_RGB')):
                os.mkdir(os.path.join(self.saver.experiment_dir, 'val_RGB'))

            val_dir = os.path.join(Path.db_root_dir(self.args.dataset), 'val')
            val_list = sorted(os.listdir(val_dir))
            val_data_path = [
                '/'.join([val_dir, val_list[i]]) for i in range(len(val_list))]

            for path in val_data_path:
                for p in path:
                    _rgb = Image.open(os.path.join(p, 'input.png')).convert('RGB')
                    _rgb = rgb_transforms(_rgb)
                    _rgb = _rgb.cpu().numpy()
                    _rgb = np.rollaxis(_rgb, 0, 3)

                    if 'val' in p:
                        plt.imsave(os.path.join(self.saver.experiment_dir,
                                                "val_RGB", f"RGB_{p.split('/')[-1]}.png"), _rgb)

        else:
            pass


def main():
    parser = segnet_argparser()
    args = parser.parse_args()
    args.cuda = torch.device('cuda:0')

    # default settings for epochs, batch_size and lr, input_channel
    if args.dataset == 'cabbage':
        args.input_channel = 3

    else:
        args.input_channel = 5

    if args.epochs is None:
        epochs = {
            'cabbage': 200,
            'cabbage5channel': 200,
        }
        args.epochs = epochs[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 16

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'cabbage': 1e-3,
            'cabbage5channel': 1e-3,
        }
        args.lr = lrs[args.dataset.lower()] / args.batch_size * args.batch_size

    if args.checkname is None:
        args.checkname = 'segnet-' + str('normal')

    print(args)
    torch.manual_seed(args.seed)
    trainer = SegNetTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)

    train_losses, val_losses = [], []

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.training(epoch)
        train_losses.append(train_loss)
        trainer.rgbsave(epoch)

        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            val_loss, save_path = trainer.validation(epoch)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                t_loss = sum(train_losses, [])
                v_loss = sum(val_losses, [])

                plt.figure(figsize=(6, 3))
                plt.plot(t_loss, color='blue', label='Train loss')
                plt.plot(v_loss, color='red', label='Validation loss')
                plt.legend()
                plt.savefig(os.path.join(save_path, "loss plot.png"))
                plt.close()


if __name__ == '__main__':
    main()
