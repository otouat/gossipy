import torch
import torch.nn as nn
import torch.nn.functional as F
from gossipy.model import TorchModel

import torch
import torch.nn as nn
from collections import OrderedDict

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 16, 3)
        self.layer2 = self.make_layer(16, 32, 3, stride=2)
        self.layer3 = self.make_layer(32, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.init_weights()

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if torch.isnan(x).any():
            print("NaN detected after conv1")
        else:
            print("No NaN detected after conv1")
        x = self.layer1(x)
        if torch.isnan(x).any():
            print("NaN detected after layer1")
        else:
            print("No NaN detected after layer1")
        x = self.layer2(x)
        if torch.isnan(x).any():
            print("NaN detected after layer2")
        else:
            print("No NaN detected after layer2")
        x = self.layer3(x)
        if torch.isnan(x).any():
            print("NaN detected after layer3")
        else:
            print("No NaN detected after layer3")
        x = self.avgpool(x)
        if torch.isnan(x).any():
            print("NaN detected after avgpool")
        else:
            print("No NaN detected after avgpool")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if torch.isnan(x).any():
            print("NaN detected after fc")
        else:
            print("No NaN detected after fc")
        return x

    def init_weights(self):  # Rename the method
        def _init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init)

    def __repr__(self) -> str:
        return "Resnet20(size=%d)" %self.get_size()

def resnet20(num_classes):
    return ResNet20(num_classes=num_classes)


class ResNet9(TorchModel):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 16, 1)
        self.layer2 = self.make_layer(16, 32, 1, stride=2)
        self.layer3 = self.make_layer(32, 64, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.init_weights()

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # Use the avgpool layer here
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def init_weights(self):  # Rename the method
        def _init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_init)

def resnet9(num_classes):
    return ResNet9(num_classes=num_classes)
