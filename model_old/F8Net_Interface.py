from .resnet import resnet18

import torchvision.models.resnet


def Model(args):
    return resnet18()
    # return torchvision.models.resnet.resnet18()