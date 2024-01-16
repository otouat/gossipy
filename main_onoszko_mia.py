import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, RandomVerticalFlip
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import DataDispatcher
from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.node import GossipNode
from gossipy.simul import GossipSimulator, AttackGossipSimulator, AttackGossipSimulator, MIAGossipSimulator, SimulationReport
from gossipy.data import get_CIFAR10
from gossipy.utils import plot_evaluation
from topology import create_torus_topology, create_simple_topology, create_circular_topology, CustomP2PNetwork

class CIFAR10Net(TorchModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def init_weights(self, *args, **kwargs) -> None:
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        #self.apply(_init_weights)
        pass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __repr__(self) -> str:
        return "CIFAR10Net(size=%d)" %self.get_size()

    def print_model_info(self):
        print("Model architecture:", self)
        print("Model parameters:")
        for name, param in self.named_parameters():
            print(f" - {name}: {param.size()}")
            print(f"{name}: {param.dtype}")

class CustomDataDispatcher(DataDispatcher):
    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = n_ex // self.n  # Ensure equal partitioning

        for idx in range(self.n):
            start_index = idx * ex_x_user
            end_index = start_index + ex_x_user
            self.tr_assignments[idx] = list(range(start_index, min(end_index, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = n_eval_ex // self.n
            for idx in range(self.n):
                start_index = idx * eval_ex_x_user
                end_index = start_index + eval_ex_x_user
                self.te_assignments[idx] = list(range(start_index, min(end_index, n_eval_ex)))

# Dataset loading
transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()
nodes_num = 18


Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]


data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte)

data_dispatcher = CustomDataDispatcher(data_handler, n=nodes_num, eval_on_user=True, auto_assign=True)

topology = create_torus_topology(9)
#topology = create_simple_topology()
#topology = create_circular_topology(10)
network = CustomP2PNetwork(topology)

nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=CIFAR10Net(),
        optimizer= torch.optim.SGD,
        optimizer_params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.001
        },
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= 256,
        local_epochs= 3),
    round_len=100,
    sync=False)

simulator = MIAGossipSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=50)

plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")