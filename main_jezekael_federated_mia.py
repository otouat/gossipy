import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode, FederatedGossipNode
from gossipy.simul import MIAGossipSimulator, MIADynamicGossipSimulator, MIAFederatedSimulator, MIASimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import get_CIFAR10, get_CIFAR100
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.utils import log_results

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()

n_classes= max(train_set[1].max().item(), test_set[1].max().item())+1
model = resnet20(n_classes)
n_nodes = 5
n_rounds = 25
n_local_epochs = 3
batch_size = 256
factors = 2
neigbors = 4
test_size=0.5
beta = 0.99
optimizer_params = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.001
}

message = f"Experiment with ResNet20 on CIFAR10 dataset (test size : {test_size}, class distribution = {beta}). {n_nodes} nodes, {n_local_epochs} local epochs, batch size {batch_size}, lr {optimizer_params['lr']}, number of neigbors {neigbors}"

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

topology = create_federated_topology(n_nodes)
network = CustomP2PNetwork(topology)

nodes = FederatedGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode = CreateModelMode.UPDATE,
        batch_size= batch_size,
        local_epochs= n_local_epochs),
    round_len=100,
    sync=False)

simulator = MIAFederatedSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PULL,
    online_prob=1,  # Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=0,  # 0.1 Simulate the possibility of message dropping,
    sampling_eval=0
)

report = MIASimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=n_rounds)

log_results(simulator, report, message)