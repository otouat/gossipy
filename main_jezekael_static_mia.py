import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher, OLDCustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import AttackGossipNode
from gossipy.simul import AttackDynamicGossipSimulator, AttackGossipSimulator, AttackSimulationReport
from gossipy.model.architecture import resnet20
from gossipy.data import get_CIFAR10
import networkx as nx
from networkx.generators import random_regular_graph
from gossipy.attacks.utils import log_results
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()

n_classes = max(train_set[1].max().item(), test_set[1].max().item()) + 1
model = resnet20(n_classes)

# Configuration dictionary
config = {
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "optimizer": "SGD",
    "architecture": "ResNet20",
    "dataset": "CIFAR-10",
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
    "echo": False,
    "ra": False
}

optimizer_params = {
    "lr": config["learning_rate"],
    "momentum": config["momentum"],
    "weight_decay": config["weight_decay"]
}

message = f"Experiment with {config['architecture']} on {config['dataset']} dataset (test size : {config['test_size']}, class distribution = {config['beta']}). | Attacks: N°Attackers: {int(config['n_nodes'] * config['p_attacker'])}, MIA: {config['mia']}, MAR: {config['mar']}, ECHO: {config['echo']} | Training: {config['n_nodes']} nodes, {config['n_local_epochs']} local epochs, batch size {config['batch_size']}, number of neigbors {config['neigbors']}, peer sampling period: {config['peer_sampling_period']} | Model: Optimizer: {config['optimizer']}, lr {config['learning_rate']},  momentum: {config['momentum']}, weight_decay: {config['weight_decay']} "

Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]

data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=config["test_size"])

data_dispatcher = OLDCustomDataDispatcher(data_handler, n=config["n_nodes"]*config["factors"], eval_on_user=True, auto_assign=True)

topology = StaticP2PNetwork(
    int(data_dispatcher.size() / config["factors"]),
    topology=nx.to_numpy_array(random_regular_graph(config["neigbors"], config["n_nodes"], seed=42))
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
        batch_size=config["batch_size"],
        local_epochs=config["n_local_epochs"]
    ),
    round_len=100,
    sync=False
)

for i, node in enumerate(nodes):
    nodes[i].mia = config["mia"]
    nodes[i].mar = config["mar"]
    if i % int(1/(config["p_attacker"])) == 0:
        nodes[i].echo = config["echo"]
        nodes[i].ra = config["ra"]

simulator = AttackGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    online_prob=1,
    drop_prob=0,
    sampling_eval=0,
    mia=config["mia"],
    mar=config["mar"],
    ra=config["ra"]
)

report = AttackSimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=config["epochs"], wall_time_limit=1.5)

log_results(simulator, report, message)