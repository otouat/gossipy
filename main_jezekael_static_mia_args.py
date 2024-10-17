import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from gossipy.core import AntiEntropyProtocol, CreateModelMode, ConstantDelay, StaticP2PNetwork
from gossipy.data import OLDCustomDataDispatcher, get_CIFAR10
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import AttackGossipNode
from gossipy.simul import AttackGossipSimulator, AttackSimulationReport
from gossipy.model.improved_resnet import resnet20
import networkx as nx
from networkx.generators import random_regular_graph
from gossipy.attacks.utils import log_results
import os
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Gossip Simulation with Attack')
    
    # Model and optimization parameters
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for the optimizer')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--architecture', type=str, default='ResNet20', help='Model architecture')
    
    # Dataset and simulation parameters
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_nodes', type=int, default=36, help='Number of nodes')
    parser.add_argument('--n_local_epochs', type=int, default=3, help='Number of local epochs')
    parser.add_argument('--neighbors', type=int, default=5, help='Number of neighbors in the network')
    parser.add_argument('--test_size', type=float, default=0.5, help='Test set size fraction')
    parser.add_argument('--factors', type=int, default=1, help='Factor to multiply nodes count for the dispatcher')
    parser.add_argument('--beta', type=float, default=0.75, help='Class distribution parameter')
    parser.add_argument('--p_attacker', type=float, default=0.3, help='Proportion of attackers')
    parser.add_argument('--send_to_all', action='store_true', default=False, help='Enable node to send its model to all his neighbors')
    
    # Attack parameters
    parser.add_argument('--mia', action='store_true', help='Enable MIA attack')
    parser.add_argument('--mar', action='store_true', help='Enable MAR attack')
    parser.add_argument('--echo', action='store_true', help='Enable ECHO attack')
    parser.add_argument('--ra', action='store_true', help='Enable RA attack')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    torch.backends.cudnn.benchmark = True
    # Setup transformations
    transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_set, test_set = get_CIFAR10()

    # Determine number of classes
    n_classes = max(train_set[1].max().item(), test_set[1].max().item()) + 1
    model = resnet20()

    optimizer_params = {
        "lr": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay
    }

    # Create message for logging
    message = f"Experiment with MIA {args.architecture} on {args.dataset} dataset (test size: {args.test_size}, class distribution = {args.beta})."
    message += f" | Attacks: N°Attackers: {int(args.n_nodes * args.p_attacker)}, MIA: {args.mia}, MAR: {args.mar}, ECHO: {args.echo}"
    message += f" | Training: {args.n_nodes} nodes, {args.n_local_epochs} local epochs, batch size {args.batch_size}, neighbors {args.neighbors}"
    message += f" | Model: Optimizer: {args.optimizer}, lr {args.learning_rate}, momentum: {args.momentum}, weight_decay: {args.weight_decay}"

    # Prepare data
    Xtr, ytr = transform(train_set[0]), train_set[1]
    Xte, yte = transform(test_set[0]), test_set[1]
    data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=args.test_size)

    # Data dispatcher
    data_dispatcher = OLDCustomDataDispatcher(data_handler, n=args.n_nodes * args.factors, eval_on_user=True, auto_assign=True)

    # Topology network
    topology = StaticP2PNetwork(
        int(data_dispatcher.size() / args.factors),
        topology=nx.to_numpy_array(random_regular_graph(args.neighbors, args.n_nodes, seed=42))
    )
    topology.compute_graph_statistics()

    # Generate nodes with attacks
    nodes = AttackGossipNode.generate(
        data_dispatcher=data_dispatcher,
        p2p_net=topology,
        model_proto=TorchModelHandler(
            net=model,
            optimizer=torch.optim.SGD,
            optimizer_params=optimizer_params,
            criterion=F.cross_entropy,
            create_model_mode=CreateModelMode.MERGE_UPDATE,
            batch_size=args.batch_size,
            local_epochs=args.n_local_epochs
        ),
        round_len=100,
        sync=False,
    )

    for i, node in enumerate(nodes):
        nodes[i].mia = args.mia
        nodes[i].mar = args.mar
        if i % int(1 / args.p_attacker) == 0:
            nodes[i].echo = args.echo
            nodes[i].ra = args.ra

    # Initialize and run the simulator
    simulator = AttackGossipSimulator(
        nodes=nodes,
        data_dispatcher=data_dispatcher,
        delta=100,
        protocol=AntiEntropyProtocol.PUSH,
        online_prob=1,
        drop_prob=0,
        sampling_eval=0,
        send_to_all=args.send_to_all,
        mia=args.mia,
        mar=args.mar,
        ra=args.ra
    )

    # Create report and start the simulation
    report = AttackSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=args.epochs, wall_time_limit=18.5)

    # Log results
    log_results(simulator, report, message)

if __name__ == "__main__":
    main()
