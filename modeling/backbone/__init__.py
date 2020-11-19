from modeling.backbone import resnet, xception, drn, mobilenet


def build_backbone(args, backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(args, output_stride, BatchNorm, pretrained=False)
    elif backbone == 'xception':
        return xception.AlignedXception(args, output_stride, BatchNorm, pretrained=False)
    elif backbone == 'drn':
        return drn.drn_d_54(args, BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
