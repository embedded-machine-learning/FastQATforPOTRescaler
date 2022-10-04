from .mobilenetv2 import MobileNetV2

def Model(args):
    return MobileNetV2()
    # return torchvision.models.resnet.resnet18()