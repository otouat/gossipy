from networkx import random_regular_graph
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
import wandb
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import FederatedAttackGossipNode, GossipNode, FederatedGossipNode
from gossipy.simul import AttackFederatedSimulator, AttackSimulationReport
from gossipy.model.architecture import *
from gossipy.model.resnet import *
from gossipy.data import get_CIFAR10, get_CIFAR100
from gossipy.topology import create_torus_topology, create_federated_topology, CustomP2PNetwork
from gossipy.attacks.federated_utils import log_results
import networkx as nx
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

wandb.init(
    project="my-awesome-project",
    config={
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "architecture": "ResNet20",
        "dataset": "CIFAR-10",
        "epochs": 250,
        "batch_size": 512,
        "n_nodes": 36,
        "n_local_epochs": 5,
        "neigbors": 5,
        "test_size": 0.7,
        "factors": 5,
        "beta": 0.99,
        "p_attacker": 1.0,
        "mia": True,
        "mar": False,
        "echo": False,
        "ra": False
    }
)

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()

n_classes = max(train_set[1].max().item(), test_set[1].max().item())+1
model = resnet20(n_classes)
wdb = wandb.config

optimizer_params = {
    "lr":  wdb.learning_rate,
    #"momentum": wdb.momentum,
    "weight_decay": wdb.weight_decay
}

message = f"Experiment with {wdb.architecture} on {wdb.dataset} dataset (test size : {wdb.test_size}, class distribution = {wdb.beta}). | Attacks: N°Attackers: {int(wdb.n_nodes * wdb.p_attacker)}, MIA: {wdb.mia}, MAR: {wdb.mar}, ECHO: {wdb.echo} | {wdb.n_nodes} nodes, {wdb.n_local_epochs} local epochs, batch size {wdb.batch_size}, lr {wdb.learning_rate}, number of neigbors {wdb.neigbors}"

Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]

data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size= wdb.test_size)

'''
assignment_method = 'label_dirichlet_skew'
assignment_params = {
    'beta': wdb.beta
}

data_dispatcher = CustomDataDispatcher(
    data_handler,
    n=wdb.n_nodes * wdb.factors,
    eval_on_user=True,
    auto_assign=False
)

# Assign data using the specified method
data_dispatcher.assign(seed=42, method=assignment_method, **assignment_params)
'''

data_dispatcher = OLDCustomDataDispatcher(data_handler, n=wdb.n_nodes*wdb.factors, eval_on_user=True, auto_assign=True)

topology = create_federated_topology(wdb.n_nodes)
network = CustomP2PNetwork(topology)

nodes = FederatedAttackGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.Adam,
        optimizer_params = optimizer_params,
        criterion = F.cross_entropy,
        create_model_mode = CreateModelMode.UPDATE,
        batch_size= wdb.batch_size,
        local_epochs= wdb.n_local_epochs),
    round_len=100,
    sync=False)

nodes[0].mia = wdb.mia
nodes[0].mar = wdb.mar
nodes[0].echo = wdb.echo

simulator = AttackFederatedSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PULL,
    online_prob=1,  # Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=0,  # 0.1 Simulate the possibility of message dropping,
    sampling_eval=0,
    mia=wdb.mia,
    mar=wdb.mar,
    ra=wdb.ra
)

report = AttackSimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=wdb.epochs, wall_time_limit=11.5)

log_results(simulator, report, wandb, message)
wandb.finish()
