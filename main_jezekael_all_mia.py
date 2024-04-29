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
from gossipy.mia.utils import log_results

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()


n_classes= max(train_set[1].max().item(), test_set[1].max().item())+1
model = ResNet152(n_classes)
n_nodes = 100
n_rounds = 200
n_local_epochs = 5
batch_size= 64
optimizer_params = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.001
    }
message = ""
model_name = model
dataset_name = "CIFAR10"

Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]


data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=0.5)

data_dispatcher = CustomDataDispatcher(data_handler, n=n_nodes, eval_on_user=True, auto_assign=True)

topology = create_torus_topology(n_nodes)
network = CustomP2PNetwork(topology)

#Static --------------------------------------
nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=model,
        optimizer= torch.optim.SGD,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= batch_size,
        local_epochs= n_local_epochs),
    round_len=100,
    sync=False)

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

log_results(simulator, report, topology, message, model_name, dataset_name)

#Dynamic --------------------------------------
nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=model,
        optimizer= torch.optim.SGD,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= batch_size,
        local_epochs= n_local_epochs),
    round_len=100,
    sync=False)

simulator = MIADynamicGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=ConstantDelay(0),
    online_prob=1,  # Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=0,  # 0.1 Simulate the possibility of message dropping,
    sampling_eval=0,
    peer_sampling_period=10
)

report = MIASimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=n_rounds)

log_results(simulator, report, topology, message, model_name, dataset_name)

#Federated --------------------------------------

topology = create_federated_topology(n_nodes)
network = CustomP2PNetwork(topology)

nodes = FederatedGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=model,
        optimizer= torch.optim.SGD,
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

log_results(simulator, report, topology, message, model_name, dataset_name)