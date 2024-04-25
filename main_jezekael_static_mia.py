import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import CustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode
from gossipy.simul import MIAGossipSimulator, MIASimulationReport
from gossipy.model.architecture import resnet20, resnet9
from gossipy.data import get_CIFAR10, get_CIFAR100
from gossipy.topology import create_torus_topology, CustomP2PNetwork
from gossipy.mia.utils import log_results

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR100()
num_nodes = 36
num_classes= max(train_set[1].max().item(), test_set[1].max().item())+1


Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]


data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte)

data_dispatcher = CustomDataDispatcher(data_handler, n=num_nodes*2, eval_on_user=True, auto_assign=True)

topology = create_torus_topology(num_nodes)
network = CustomP2PNetwork(topology)

nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=resnet9(num_classes),
        optimizer= torch.optim.SGD,
        optimizer_params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.001
        },
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= 256,
        local_epochs= 5),
    round_len=100,
    sync=False)

simulator = MIAGossipSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0,
)

report = MIASimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=50)

log_results(simulator, report, topology)