import torch
import torch.nn as nn
import torch.nn.functional as F
from gossipy.model import TorchModel


class ResNet20(TorchModel):
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
        log = False
        if torch.isnan(x).any() and log:
            print("NaN detected after conv1")
        x = self.layer1(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer1")
        x = self.layer2(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer2")
        x = self.layer3(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer3")
        x = self.avgpool(x)  # Use the avgpool layer here
        if torch.isnan(x).any() and log:
            print("NaN detected after avgpool")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after fc")
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
        return "Resnet20"

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

class ResNet50(TorchModel):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=3)
        self.layer2 = self._make_layer(64, 128, num_blocks=4)
        self.layer3 = self._make_layer(128, 256, num_blocks=6)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256*4, 256)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            ))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, kernel_size=7)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x

    def init_weights(self):  
        def _init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_init)

def resnet50(num_classes):
    return ResNet50(num_classes=num_classes)

#-------------------------------------------------- TESTING ---------------------------------------------------#

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

class NewResNet20(TorchModel):
    def __init__(self, num_classes=10):
        super(NewResNet20, self).__init__()
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
        layers.append(BasicBlock(in_channels, out_channels, stride))  # CHANGED
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))  # CHANGED
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        log = False
        if torch.isnan(x).any() and log:
            print("NaN detected after conv1")
        x = self.bn1(x)  # ADDED
        x = self.layer1(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer1")
        x = self.layer2(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer2")
        x = self.layer3(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after layer3")
        x = self.avgpool(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after avgpool")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if torch.isnan(x).any() and log:
            print("NaN detected after fc")
        return x  # Softmax can be applied during loss computation

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
        return "Resnet20"

def Newresnet20(num_classes):
    return NewResNet20(num_classes=num_classes)
