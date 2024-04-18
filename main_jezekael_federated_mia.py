import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, RandomVerticalFlip
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import FederatedGossipNode
from gossipy.simul import MIAFederatedSimulator, SimulationReport
from gossipy.model.architecture import resnet20
from gossipy.data import get_CIFAR10, get_CIFAR100
from topology import create_federated_topology, display_topology,  CustomP2PNetwork
from gossipy.MIA.mia import plot_mia_vulnerability, log_results, get_fig_evaluation

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
train_set, test_set = get_CIFAR100()
nodes_num = 16
num_classes = max(train_set[1].max().item(), test_set[1].max().item())+1


Xtr, ytr = transform(train_set[0]), train_set[1]
Xte, yte = transform(test_set[0]), test_set[1]


data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte)

data_dispatcher = CustomDataDispatcher(data_handler, n=nodes_num, eval_on_user=True, auto_assign=True)

topology = create_federated_topology(nodes_num)
network = CustomP2PNetwork(topology)

nodes = FederatedGossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=network,
    model_proto=TorchModelHandler(
        net=resnet20(num_classes),
        optimizer= torch.optim.SGD,
        optimizer_params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.001
        },
        criterion = F.cross_entropy,
        create_model_mode = CreateModelMode.UPDATE,
        batch_size= 256,
        local_epochs= 20),
    round_len=100,
    sync=False)

simulator = MIAFederatedSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PULL,
    sampling_eval=0
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=100)

fig = get_fig_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")
fig2, fig3 = plot_mia_vulnerability(simulator.mia_accuracy, simulator.gen_error)
fig4 = display_topology(topology)
diagrams = {
    'Overall test results': fig,
    'mia_vulnerability over Gen error': fig2,
    'mia_vulnerability over epoch': fig3,
    "Topology": fig4
}
log_results(simulator, simulator.n_rounds, diagrams, report.get_evaluation(False))