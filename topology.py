import random
import math

def create_torus_topology(size):
    assert math.sqrt(size) == int(math.sqrt(size)), "Size must be a perfect square for a torus"
    dim = int(math.sqrt(size))
    topology = {}

    for node in range(size):
        x, y = divmod(node, dim)
        neighbors = [
            ((x + 1) % dim) * dim + y,  # Down
            ((x - 1) % dim) * dim + y,  # Up
            x * dim + (y + 1) % dim,    # Right
            x * dim + (y - 1) % dim     # Left
        ]
        topology[node] = neighbors

    return topology


def create_social_topology(size, min_degree, max_degree):
    topology = {}
    all_nodes = list(range(size))

    for node in range(size):
        degree = random.randint(min_degree, max_degree)
        neighbors = random.sample(all_nodes, degree)
        if node in neighbors:  # Remove self-connections
            neighbors.remove(node)
        topology[node] = neighbors

    return topology

def create_simple_topology():
    topology = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1]
    }
    return topology

def create_circular_topology(size):
    topology = {}
    for node in range(size):
        prev_node = (node - 1) % size  # Get previous node index
        next_node = (node + 1) % size  # Get next node index
        topology[node] = [prev_node, next_node]  # Connect each node to previous and next
    return topology

def create_federated_topology(size):
    topology = {}
    all_nodes = list(range(1, size))  # Exclude node 0 from all_nodes

    # Create a hub node that is connected to all other nodes
    hub_node = 0
    all_nodes.append(hub_node)

    for node in range(1, size):  # Start from 1 instead of 0
        topology[node] = [hub_node]  # Each node is connected only to the hub node

    # Connect the hub node to all other nodes
    topology[hub_node] = list(range(1, size))  # Start from 1 instead of 0

    return topology

class CustomP2PNetwork:
    def __init__(self, topology):
        self.topology = topology

    def get_peers(self, node_index):
        # Get the indices of neighbor nodes
        return self.topology.get(node_index, [])

    def size(self):
        return len(self.topology)
    
import networkx as nx
import matplotlib.pyplot as plt

def display_topology(topology):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for node in topology:
        G.add_node(node)

    # Add edges
    for node, neighbors in topology.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)

    plt.title("Topology")
    fig = plt.gcf()
    return fig