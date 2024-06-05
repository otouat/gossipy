import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork, UniformDynamicP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import AttackGossipNode, GossipNode, FederatedGossipNode
from gossipy.simul import AttackDynamicGossipSimulator, AttackSimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import get_CIFAR10, get_CIFAR100
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.utils import log_results
import networkx as nx
from networkx.generators import random_regular_graph

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()

n_classes = max(train_set[1].max().item(), test_set[1].max().item())+1
model = newresnet20(n_classes)
# Parameters:
n_nodes = 100
n_rounds = 250
n_local_epochs = 3
batch_size = 256
factors = 1
neigbors = 4
test_size=0.5
beta = 0.99
p_attacker = 1.0
# Attacks:
mia = True
mar = False
echo = False
ra = False
peer_sampling_period=5
optimizer_params = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.001
    }

message = f"Experiment with ResNet20 on CIFAR10 dataset (test size : {test_size}, class distribution = {beta}). | Attacks: N°Attackers: {int(n_nodes*p_attacker)}, MIA: {mia}, MAR: {mar}, ECHO: {echo} | {n_nodes} nodes, {n_local_epochs} local epochs, batch size {batch_size}, lr {optimizer_params['lr']}, number of neigbors {neigbors}, and peer sampling period {peer_sampling_period}"

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

topology = UniformDynamicP2PNetwork(int(data_dispatcher.size()/factors), topology=nx.to_numpy_array(random_regular_graph(neigbors, n_nodes, seed=42)))

nodes = AttackGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= batch_size,
        local_epochs= n_local_epochs),
    round_len=100,
    sync=False)

for i in range(1, n_nodes):
    nodes[i].mia = mia
    nodes[i].mar = mar
    if i % int(1/(p_attacker)) == 0:
        nodes[i].echo = echo

simulator = AttackDynamicGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=ConstantDelay(0),
    online_prob=1,  # Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=0,  # 0.1 Simulate the possibility of message dropping,
    sampling_eval=0,
    peer_sampling_period=peer_sampling_period,
    mia=mia,
    mar=mar,
    ra=ra
)

report = AttackSimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=n_rounds, wall_time_limit=11)

log_results(simulator, report, message)