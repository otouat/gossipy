import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import ImprovedTorchModelHandler, TorchModelHandler
from gossipy.node import GossipNode, FederatedGossipNode, AttackGossipNode
from gossipy.simul import MIAGossipSimulator, MIADynamicGossipSimulator, MIAFederatedSimulator, MIASimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import *
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.utils import log_results
import networkx as nx
from networkx.generators import random_regular_graph
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()

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
    
n_classes = max(train_set[1].max().item(), test_set[1].max().item())+1
model = resnet20(n_classes)
n_nodes = 10
n_rounds = 25
n_local_epochs = 3
batch_size = 256
factors = 10
neigbors = 4
test_size=0.5
beta = 0.99
p_attacker = 1.0
mia = True
mar = False
echo = False
optimizer_params = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.001
}
scheduler = lr_scheduler.StepLR  # Choosing a StepLR scheduler
scheduler_params = {'step_size': 30, 'gamma': 0.1}

message = f"Experiment with ResNet20 on CIFAR10 dataset (test size : {test_size}, class distribution = {beta}). | Attacks: N°Attackers: {int(n_nodes*p_attacker)}, MIA: {mia}, MAR: {mar}, ECHO: {echo} | {n_nodes} nodes, {n_local_epochs} local epochs, batch size {batch_size}, lr {optimizer_params['lr']}, number of neigbors {neigbors}"

Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]

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

#data_dispatcher = OLDCustomDataDispatcher(data_handler, n=n_nodes*factors, eval_on_user=True, auto_assign=True)

topology = StaticP2PNetwork(int(data_dispatcher.size()/factors), topology=nx.to_numpy_array(random_regular_graph(neigbors, n_nodes, seed=42)))

nodes = AttackGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=ImprovedTorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= batch_size,
        local_epochs= n_local_epochs,
        scheduler=scheduler,
        scheduler_params=scheduler_params),
    round_len=100,
    sync=False)

for i in range(1, n_nodes):
    if i % int(1/(p_attacker)) == 0:
        nodes[i].mar = True
        nodes[i].mar = False
        nodes[i].echo = True

simulator = MIAGossipSimulator(
    nodes = nodes,
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

log_results(simulator, report, message)