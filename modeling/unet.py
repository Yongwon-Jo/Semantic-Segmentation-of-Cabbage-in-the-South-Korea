
from __future__ import print_function

import sys
sys.path.append("D:/PROJECT/LX_DeepLabV3+_v2")

from collections import OrderedDict

from utils.layers import init_weights


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pprint


# copy and crop
class CopyCrop(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CopyCrop, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2):
        up_result = self.up(x1)

        crop_row = (x2.shape[2] - up_result.shape[2]) // 2 - 1
        crop_col = up_result.shape[2] + crop_row

        x2_final = x2[:, :, crop_row: crop_col, crop_row: crop_col]
        x = torch.cat([x2_final, up_result], dim=1)

        return x


# unet
class UNet(nn.Module):
    def __init__(self, args, num_classes=2):
        super(UNet, self).__init__()
        self.args = args

        # Encoder
        self.encoder_block = [2, 2, 2, 2, 2]  # num of blocks
        self.encoder_kernel_num = [self.args.input_channel, 64, 128, 256, 512, 1024]

        for i in range(len(self.encoder_block)):
            block = []

            if i == 0:
                pass
            else:
                block.append(nn.MaxPool2d(2, stride=2))

            for j in range(self.encoder_block[i]):
                if j == 0:
                    block.append(nn.Conv2d(self.encoder_kernel_num[i],
                                           self.encoder_kernel_num[i + 1],
                                           3, stride=1, padding=0))
                else:
                    block.append(nn.Conv2d(self.encoder_kernel_num[i + 1],
                                           self.encoder_kernel_num[i + 1],
                                           3, stride=1, padding=0))
                block.append(nn.BatchNorm2d(self.encoder_kernel_num[i + 1]))
                block.append(nn.ReLU(inplace=True))

            setattr(self, 'encoder_block' + str(i + 1), nn.Sequential(*block))
            self.apply(init_weights)

        # Decoder
        self.decoder_block = [2, 2, 2, 2]
        self.decoder_kernel_num = [1024, 512, 256, 128, 64, num_classes]

        for i in range(len(self.decoder_block)):
            block = []
            block_conv = []

            block_conv.append(CopyCrop(self.decoder_kernel_num[i], self.decoder_kernel_num[i + 1]))

            for j in range(self.decoder_block[i]):
                if j != self.decoder_block[i] - 1:
                    block.append(nn.Conv2d(self.decoder_kernel_num[i],
                                           self.decoder_kernel_num[i],
                                           3, stride=1, padding=0))
                    block.append(nn.BatchNorm2d(self.decoder_kernel_num[i]))
                else:
                    block.append(nn.Conv2d(self.decoder_kernel_num[i],
                                           self.decoder_kernel_num[i + 1],
                                           3, stride=1, padding=0))
                    block.append((nn.BatchNorm2d(self.decoder_kernel_num[i + 1])))
                block.append(nn.ReLU(inplace=True))

            if i == len(self.decoder_block) - 1:
                block.append(nn.Conv2d(self.decoder_kernel_num[i + 1], num_classes,
                                       1, stride=1))

            setattr(self, 'decoder_upconv_block' + str(len(self.decoder_block) - i), *block_conv)
            setattr(self, 'decoder_block' + str(len(self.decoder_block) - i), nn.Sequential(*block))

    def forward(self, x):

        # encoder
        input_size = x.size()

        xe1 = self.encoder_block1(x)
        xe1_size = xe1.size()
        # print(xe1_size)

        xe2 = self.encoder_block2(xe1)
        xe2_size = xe2.size()
        # print(xe2_size)

        xe3 = self.encoder_block3(xe2)
        xe3_size = xe3.size()
        # print(xe3_size)

        xe4 = self.encoder_block4(xe3)
        xe4_size = xe4.size()
        # print(xe4_size)

        xe5 = self.encoder_block5(xe4)
        xe5_size = xe5.size()
        # print(xe5_size)

        # decode
        xd4 = self.decoder_upconv_block4(xe5, xe4)
        xd4 = self.decoder_block4(xd4)
        # print(xd4.size())

        xd3 = self.decoder_upconv_block3(xd4, xe3)
        xd3 = self.decoder_block3(xd3)
        # print(xd3.size())

        xd2 = self.decoder_upconv_block2(xd3, xe2)
        xd2 = self.decoder_block2(xd2)
        # print(xd2.size())

        xd1 = self.decoder_upconv_block1(xd2, xe1)
        xd1 = self.decoder_block1(xd1)
        # print(xd1.size())

        return xd1


if __name__=='__main__':
    from utils.args import unet_argparser
    import numpy as np

    parser = unet_argparser()
    args = parser.parse_args([])
    args.input_channel = 5

    model = UNet(args, 2)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    

    print(params)