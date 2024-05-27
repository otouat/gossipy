import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomCrop, ToTensor
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, UniformDynamicP2PNetwork
from gossipy.data import CustomDataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode
from gossipy.simul import MIADynamicGossipSimulator, MIASimulationReport
from gossipy.model.resnet import resnet20
from gossipy.data import get_CIFAR10
from networkx.generators import random_regular_graph
import networkx as nx
from gossipy.attacks.utils import log_results

# Data augmentation and normalization
transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_test = Compose([
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset
train_set, test_set = get_CIFAR10()

# Apply transformations
Xtr, ytr = transform_train(train_set[0]), train_set[1]
Xte, yte = transform_test(test_set[0]), test_set[1]

# Determine the number of classes
n_classes = max(ytr.max().item(), yte.max().item()) + 1

# Initialize model
model = resnet20(n_classes)

# Simulation parameters
n_nodes = 100
n_rounds = 150
n_local_epochs = 5
batch_size = 256
factors = 1
neighbors = 4
peer_sampling_period = 10

# Optimizer parameters
optimizer_params = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.001
}

# Log message
message = f"Experiment with ResNet20 on CIFAR10 dataset. {n_nodes} nodes, {n_local_epochs} local epochs, batch size {batch_size}, lr {optimizer_params['lr']}, number of neighbors {neighbors}, and peer sampling period {peer_sampling_period}"

# Prepare data handler
data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=0.5)

# Data assignment method
assignment_method = 'label_dirichlet_skew'
assignment_params = {
    'beta': 0.99
}

# Custom data dispatcher
data_dispatcher = CustomDataDispatcher(
    data_handler,
    n=n_nodes * factors,
    eval_on_user=True,
    auto_assign=False
)

# Assign data using the specified method
data_dispatcher.assign(seed=42, method=assignment_method, **assignment_params)

# Create network topology
topology = UniformDynamicP2PNetwork(int(data_dispatcher.size() / factors), topology=nx.to_numpy_array(random_regular_graph(neighbors, n_nodes, seed=42)))

# Generate nodes
nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=TorchModelHandler(
        net=model,
        optimizer=torch.optim.SGD,
        optimizer_params=optimizer_params,
        criterion=F.cross_entropy,
        create_model_mode=CreateModelMode.MERGE_UPDATE,
        batch_size=batch_size,
        local_epochs=n_local_epochs
    ),
    round_len=100,
    sync=False
)

# Initialize and start the simulator
simulator = MIADynamicGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=ConstantDelay(0),
    online_prob=1,
    drop_prob=0,
    sampling_eval=0,
    peer_sampling_period=peer_sampling_period
)

report = MIASimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=n_rounds)

# Log results
log_results(simulator, report, message)
