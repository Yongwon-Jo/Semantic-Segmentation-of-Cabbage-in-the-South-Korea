import argparse


def deeplab_argparser():
    parser = argparse.ArgumentParser(description='PyTorch DeeplabV3Plus Training')
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception'],
                        help='backbone name (default: xception)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cabbage5channel',
                        choices=['cabbage', 'cabbage5channel'],
                        help='dataset name (default: cabbage)')
    parser.add_argument('--input-channel', type=int, default='3',
                        help='input data size (default: 3)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        metavar='M', help='weight decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # fine-tuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='fine-tuning on a different dataset')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # choose model
    parser.add_argument('--models', type=str, default='DeepLab',
                        choices=['DeepLab', 'SegNet', 'UNet'],
                        help='Choose models (default: DeepLab)')

    return parser


def unet_argparser():
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')
    parser.add_argument('--dataset', type=str, default='cabbage5channel',
                        choices=['cabbage', 'cabbage5channel'],
                        help='dataset name (default: cabbage)')
    parser.add_argument('--input-channel', type=int, default='3',
                        help='input data size (default: 3)')
    parser.add_argument('--base-size', type=int, default=388,
                        help='base image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.99,
                        metavar='M', help='momentum (default: 0.99)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # choose model
    parser.add_argument('--models', type=str, default='UNet',
                        choices=['DeepLab', 'SegNet', 'UNet'],
                        help='Choose models (default: UNet)')

    return parser


def segnet_argparser():
    parser = argparse.ArgumentParser(description='PyTorch SegNet Training')
    parser.add_argument('--dataset', type=str, default='cabbage',
                        choices=['cabbage', 'cabbage5channel'],
                        help='dataset name (default: cabbage)')
    parser.add_argument('--input-channel', type=int, default='3',
                        help='input data size (default: 3)')
    parser.add_argument('--base-size', type=int, default=224,
                        help='base image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # choose model
    parser.add_argument('--models', type=str, default='SegNet',
                        choices=['DeepLab', 'SegNet', 'UNet'],
                        help='Choose models (default: SegNet)')

    return parser
