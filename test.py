import wandb
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode, FederatedGossipNode, AttackGossipNode
from gossipy.simul import AttackGossipSimulator, AttackSimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import get_FEMNIST
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.utils import log_results
import networkx as nx
from networkx.generators import random_regular_graph
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

import torch
import torch.nn as nn
import torch.nn.functional as F
from gossipy.model import TorchModel

class ResNet20(TorchModel):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
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

wandb.init(
    project="my-awesome-project",
    config={
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "optimizer": "SGD",
        "architecture": "ResNet20",
        "dataset": "FEMNIST",
        "epochs": 250,
        "batch_size": 256,
        "n_nodes": 36,
        "n_local_epochs": 3,
        "neigbors": 5,
        "test_size": 0.5,
        "factors": 1,
        "beta": 0.99,
        "p_attacker": 0.3,
        "mia": True,
        "mar": True,
        "echo": True,
        "ra": False
    }
)

# Load FEMNIST dataset
train_data, test_data = get_FEMNIST()

# Print the structure of train_data and test_data
print("Structure of train_data:", type(train_data), len(train_data), type(train_data[0]), len(train_data[0]))
print("Structure of test_data:", type(test_data), len(test_data), type(test_data[0]), len(test_data[0]))

# Unpack the dataset tuples
Xtr, ytr, _ = train_data
Xte, yte, _ = test_data

# Ensure the input data is in the correct shape: [batch_size, 1, 28, 28]
Xtr = Xtr.view(-1, 1, 28, 28)
Xte = Xte.view(-1, 1, 28, 28)

# Determine the number of classes
n_classes = max(ytr.max().item(), yte.max().item()) + 1

# Initialize the model
model = resnet20(n_classes)
wdb = wandb.config

optimizer_params = {
    "lr": wdb.learning_rate,
    "momentum": wdb.momentum,
    "weight_decay": wdb.weight_decay
}

message = f"Experiment with {wdb.architecture} on {wdb.dataset} dataset (test size: {wdb.test_size}, class distribution = {wdb.beta}). | Attacks: N°Attackers: {int(wdb.n_nodes * wdb.p_attacker)}, MIA: {wdb.mia}, MAR: {wdb.mar}, ECHO: {wdb.echo} | Training: {wdb.n_nodes} nodes, {wdb.n_local_epochs} local epochs, batch size {wdb.batch_size}, number of neigbors {wdb.neigbors} | Model: Optimizer: {wdb.optimizer}, lr {wdb.learning_rate}, momentum: {wdb.momentum}, weight_decay: {wdb.weight_decay}"

data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=wdb.test_size)

data_dispatcher = OLDCustomDataDispatcher(data_handler, n=wdb.n_nodes * wdb.factors, eval_on_user=True, auto_assign=True)

topology = StaticP2PNetwork(
    int(data_dispatcher.size() / wdb.factors),
    topology=nx.to_numpy_array(random_regular_graph(wdb.neigbors, wdb.n_nodes, seed=42))
)
nodes = AttackGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params=optimizer_params,
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=wdb.batch_size,
        local_epochs=wdb.n_local_epochs),
    round_len=100,
    sync=False)

print(f"Number of nodes generated: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"Node {i} created")

for i, node in enumerate(nodes):
    nodes[i].mia = wdb.mia
    nodes[i].mar = wdb.mar
    if i % int(1 / (wdb.p_attacker)) == 0:
        nodes[i].echo = wdb.echo
        nodes[i].ra = wdb.ra

simulator = AttackGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    online_prob=1,
    drop_prob=0,
    sampling_eval=0,
    mia=wdb.mia,
    mar=wdb.mar,
    ra=wdb.ra
)

report = AttackSimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=wdb.epochs, wall_time_limit=10.5)

log_results(simulator, report, wandb, message)
wandb.finish()
