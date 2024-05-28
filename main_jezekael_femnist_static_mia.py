import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode, FederatedGossipNode, AttackGossipNode
from gossipy.simul import MIAGossipSimulator, MIADynamicGossipSimulator, MIAFederatedSimulator, MIASimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import *
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.utils import log_results
import networkx as nx
from networkx.generators import random_regular_graph
import numpy as np
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Define the ResNet20 model
class ResNet20(TorchModel):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Change input channels to 1
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
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
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

def resnet20(num_classes):
    return ResNet20(num_classes=num_classes)
    
# Correct the normalization for grayscale images
transform = Compose([Normalize([0.5], [0.5])])

# Load the FEMNIST dataset
train_set, test_set = get_FEMNIST()

# Confirm the shapes of the datasets
print(train_set[0].shape)  # Should output (N, 28, 28)
print(test_set[0].shape)   # Should output (M, 28, 28)

# Add an unsqueeze step to convert to (N, 1, 28, 28)
Xtr = train_set[0].unsqueeze(1)
Xte = test_set[0].unsqueeze(1)

print(Xtr.shape)  # Should output (N, 1, 28, 28)
print(Xte.shape)  # Should output (M, 1, 28, 28)

# Apply normalization transform
Xtr = torch.stack([transform(image) for image in Xtr])
Xte = torch.stack([transform(image) for image in Xte])

print(Xtr.shape)  # Should output (N, 1, 28, 28)
print(Xte.shape)  # Should output (M, 1, 28, 28)

# Convert target tensor to torch.long
ytr = train_set[1].long()
yte = test_set[1].long()

n_classes = max(ytr.max().item(), yte.max().item()) + 1
model = resnet20(n_classes)
n_nodes = 100
n_rounds = 250
n_local_epochs = 5
batch_size = 256
factors = 1
neigbors = 4
test_size=0.5
beta = 0.99
optimizer_params = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.001
}

message = f"Experiment with ResNet20 on FEMNIST dataset (test size : {test_size}, class distribution = {beta}). {n_nodes} nodes, {n_local_epochs} local epochs, batch size {batch_size}, lr {optimizer_params['lr']}, number of neigbors {neigbors}"

data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=test_size)

assignment_method = 'label_dirichlet_skew'
assignment_params = {
    'beta': beta
}

data_dispatcher = CustomDataDispatcher(
    data_handler,
    n=n_nodes * factors,
    eval_on_user=True,
    auto_assign=False
)

# Assign data using the specified method
data_dispatcher.assign(seed=42, method=assignment_method, **assignment_params)

topology = StaticP2PNetwork(int(data_dispatcher.size() / factors), topology=nx.to_numpy_array(random_regular_graph(neigbors, n_nodes, seed=42)))

nodes = AttackGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params=optimizer_params,
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=batch_size,

        local_epochs=n_local_epochs),
    round_len=100,
    sync=False)

simulator = MIAGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    online_prob=1,  # Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=0,  # 0.1 Simulate the possibility of message dropping,
    sampling_eval=0,
)

report = MIASimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=n_rounds)
