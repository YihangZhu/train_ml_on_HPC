import torch.nn as nn

from models import resnet_cifar


class ResNet32(nn.Module):
    def __init__(self, num_blocks, num_classes, input_channels=3):
        super().__init__()
        self.resnet = resnet_cifar.ResNet_Cifar(resnet_cifar.BasicBlock, [num_blocks, num_blocks, num_blocks],
                                                input_channels=input_channels)
        self.fc = nn.Linear(self.resnet.in_planes, num_classes)
        self.apply(resnet_cifar.weights_init)

    def forward(self, x, get_feature=False):
        out = self.resnet(x)
        # from utils.visualize import plot_feature_map, plot_image
        # for ind, image in enumerate(out):
        #     plot_feature_map(x[ind], 3, 1, None)
        #     plot_feature_map(image, 8, 8, None)
        out = nn.functional.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        # if torch.min(feature) < 0:
        #     print(f"a: {feature}\n{torch.min(feature)}")
        #     raise Exception('feature contain negative a')
        out = self.fc(feature)
        if get_feature:
            return out, feature
        return out


# 2n+2n+2n+2 = 6n+2
# each res unit has two lays, each network has three types of res units regarding dimension, 16, 32, 64
# (16+32+64)*3*2*3*3 + 16*3*3 + 64*10 + 10 = 6842
def resnet32(**kwargs):
    return ResNet32(num_blocks=5, **kwargs)


def resnet44(**kwargs):
    return ResNet32(num_blocks=9, **kwargs)


def resnet110(**kwargs):
    return ResNet32(num_blocks=18, **kwargs)


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def to(self, device):
        self.ce.to(device)
        return self

    def forward(self, output_logits, target):
        loss = self.ce(output_logits, target)
        return loss, output_logits
