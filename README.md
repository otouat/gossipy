# Getting started

## USAGE

```console
!git clone https://github.com/Jezekael/gossipy.git
%cd gossipy
!pip install -r requirements.txt
```

## Introduction

**gossipy** is a python module created for research purposes and designed to be used by researchers
and developers. It is thought to be easy-to-use and easy-to-extend.

This framework provides a set of tools for simulating gossip learning and decentralized 
federated learning algorithms. 

## Features

- [x] Loss/Enthropy Threshold based Membership Inference Attack
- [x] Classed specific Threshold
- [x] Marginalized Membership Inference Attack
- [x] Echo Attack
- [x] Decentralized Federated Learning
- [x] non i.i.d DataDispatcher

## TODOs
- [ ] Echo Attack with Marginalized Model
- [ ] Reconstruction Attack
- [ ] Implementation of NICO++

### Gossip Learning

Gossip Learning (GL) is a distributed (machine) learning methodology based on the idea of spreading
information to other nodes without the need of a centralized server. The name *gossip* is a reference
to the gossiping mechanism used in ad-hoc networks.

The goal of GL is to train a machine learning model on a data set distributed across several nodes
in a network while keeping the local data private.

In its simplest form, that we call here *vanilla*, the gossip learning framework can be summarized
as in the following algorithm:

```{figure} ./imgs/gl_framework_small.png
:height: 260px
:name: gossip_learning_alg

The *vanilla* Gossip Learning algorithm.
```

The algorithm shows what happen on a single node, and we can highlight three important macro steps:
1.  **model initialization**: (line 1-2) the model is initialized and (potentially) trained on the
local data. This is a one time procedure;
2.  **gossiping**: (line 3-6) the model is repeatedly gossiped to other, usually selected at random,
nodes;
3.  **model update**: (line 7-9) given a just received model, the model is trained on the local data
and set as the new model. Usually, the model update is done by merging the local model with the
received model (merge-update). However, in it simplest form, the model update is done by replacing
the local model with the received model (replace-update).

## Setting up a simulation

**gossipy** allows to define a gossip learning simulation easily and quickly. The four main
components that needs to be set up are:
- a [DataHandler](): this is the object responsible to handle the dataset.
After the dataset is loaded in memory, the data handler manages the splitting of the dataset
into training and test sets:
- a [DataDispatcher](): this is the object responsible to dispatch the data across
the clients (i.e., the nodes of the network).
- [GossipNodes](): these are the nodes participating in the simulation. Each node own a model
that is created starting from a [ModelHandler]() prototype. The [ModelHandler]() is the object
responsible to handle the model training and evaluation.
- a [GossipSimulation](): this is the main object that runs the gossip learning
simulation. 


The following code snippets show how to set up a gossip learning simulation.

```{note}
The following example tries to reproduce one of the experiments reported in the paper

Ormándi, Róbert, István Hegedüs, and Márk Jelasity. ['Gossip Learning with Linear Models on
Fully Distributed Data'](https://doi.org/10.1002/cpe.2858). Concurrency and Computation:
Practice and Experience 25, no. 4 (February 2013): 556–571.
```

Let's start by loading the dataset.

```python
from gossipy.data import AssignmentHandler, NonIIDCustomDataDispatcher, OLDCustomDataDispatcher, get_CIFAR10
from gossipy.data.handler import ClassificationDataHandler

transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_set, test_set = get_CIFAR10()
```

Here, we loaded `CIFAR10` dataset as PyTorch tensors. Now, we need a `DataHandler` and a
`DataDispatcher` to manage the distribution of the dataset across the nodes.

```python
data_handler = ClassificationDataHandler(Xtr, ytr, Xte, yte, test_size=0.5)

data_dispatcher = OLDCustomDataDispatcher(data_handler, n=10, eval_on_user=True, auto_assign=True)
```

We fixed the test size to 10% of the dataset and we perform global evaluation on a separate test 
set, i.e., the nodes do not have local test sets. The number of nodes in the network is 100.
The `auto_assign` parameter is set to `True` because we want just to compute the random
assignment of the dataset to the nodes.

Gossip learning is based on peer-to-peer communication. Therefore, we need a to define the topology 
of the network. In this case, we use a fully connected network, thus a clique topology.

```python
topology = StaticP2PNetwork(num_nodes=data_dispatcher.size(), topology=None)
```

It is called *static* because the topology is fixed throughout the simulation. 

```python
topology =  UniformDynamicP2PNetwork(data_dispatcher.size(), topology=nx.to_numpy_array(random_regular_graph(config["neigbors"], config["n_nodes"], seed=42)))
```
It is called *Dynamic* because the topology moves throughout the simulation depending on peer_sampling_period. 


## Nodes
The AttackGossipNode class extends the GossipNode class to include functionalities 
specifically designed for performing membership inference attacks in a peer-to-peer (P2P) network. 
The nodes holds the model and the data.
Here’s an overview of how it works and what it does:
1. Model Initialization
2. Peer Selection
3. Sending and Receiving Messages
4. Calls Model Evaluation
 

```python

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
```

AttackGossipNode are created using the [GossipNode.generate()]() method which 
generates a number of nodes (corresponding to the size of the network/assignment of the dispatcher)
starting from a prototype. The prototype is actually the model handler that represents the crucial
information of the node. Nodes are created with an incremental id starting from 0. The `sync` 
paramenter controls whether the nodes are synchronized or not. Synchronized nodes participate
exactly once every each round.

Now, we can set up the simulation: we set a round length of 100 time units, a PUSH propocol (i.e.,
nodes only send their model to other nodes), and we set to 10% the number of nodes considered during
each evaluation.

```python
simulator = AttackGossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    online_prob=1,
    drop_prob=0.2,
    sampling_eval=0,
    mia=config["mia"],
    mar=config["mar"],
    ra=config["ra"]
)
```

## Simulation
The Attacksimulation class coordinates all the different branches together. To keep track of the progress of the simulation, we use a
[ AttackSimulationReport]() object that is attached to the simulation. A `SimulationReport` collects the
results and some useful information about the simulation (e.g., number of sent messages,
number of failed messages, and the attacks results...).

```python
report = AttackSimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=100, wall_time_limit=18.5)
```

Once the simulation is finished, we can access the report object and, for example, plot the 
results.

```python
plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")
```

## Membership Inference Attack

Overview
The Membership Inference Attack (MIA) is a privacy attack where the adversary aims to determine
whether a particular data point was part of the training dataset of a model. This attack leverages
the differences in the model's behavior on training versus unseen data points. 
In the context of the gossipy framework, MIA is implemented to evaluate the privacy risks 
associated with the models trained in a gossip learning setting.


```python
mia_for_each_nn(simulation, attackerNode)
```
This function is called every round with 3 possible states:
- standart mia:
- class-based mia: This function evaluates the class-specific
performance for the membership inference attack, providing thresholds for each class.
 (more efficient and precise, usefull for non i.i.d setting)
- marganlized mia: the attack isolates the model of its victim from the local updates of all its neighboors


```python
def mia_best_th(model, train_data, test_data, device, nt=200)
```
This function computes the best threshold for membership inference based on loss and entropy,
with nt representing the number of points on linear space to test the differents threshold 
between the minimum loss and maximim loss
