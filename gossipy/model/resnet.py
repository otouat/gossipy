import torch
import torch.nn as nn
from gossipy.model import TorchModel

class BasicBlock(nn.Module):
    """Basic Block for ResNet 18 and ResNet 34"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    """Residual block for ResNet over 50 layers"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        else:
            self.shortcut = nn.Identity()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x

class TorchResNet(TorchModel):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    @torch.autocast(device_type="cuda")
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_init)

class ResNet18(TorchResNet):
    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes)

class ResNet34(TorchResNet):
    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes)

class ResNet50(TorchResNet):
    def __init__(self, num_classes=100):
        super().__init__(BottleNeck, [3, 4, 6, 3], num_classes)

class ResNet101(TorchResNet):
    def __init__(self, num_classes=100):
        super().__init__(BottleNeck, [3, 4, 23, 3], num_classes)

class ResNet152(TorchResNet):
    def __init__(self, num_classes=100):
        super().__init__(BottleNeck, [3, 8, 36, 3], num_classes)
