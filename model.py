import torch.nn as nn
from torchvision.models import resnet18


def build_backbone():
    """ResNet-18 adapted for 32x32 inputs. Outputs a 512-d feature vector.

    The stock torchvision stem (7x7 stride-2 conv + maxpool) throws away too much
    spatial detail on 32x32 images, so we swap in the standard CIFAR stem.
    """
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()  # backbone now returns 512-d features
    return net


class Head(nn.Module):
    """A detachable linear classifier. The authorized task keeps one of these;
    the simulated attacker attaches a fresh one each suppression round."""

    def __init__(self, num_classes=10, in_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
