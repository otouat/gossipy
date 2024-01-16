import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def mia_for_each_nn(nodes, attackerNode, data_dispatcher):
    idx = attackerNode.idx
    nn = sorted(attackerNode.p2p_net.get_peers(idx), key=lambda x: int(x))
    model = copy.deepcopy(attackerNode.model_handler.model)
    mias = np.zeros((len(nn), 3))
    i = 0
    for node in nodes.values():  # Iterate through node instances, not indices
      #print(f"Is node n°{node.idx} in neigboors: {nn}? {node.idx in nn}")
      if node.idx in nn:
        data = node.data
        train_data, test_data = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        r = mia_best_th(model, train_data, test_data, device)
        #print(f"results: {r}")
        mias[i] = r
        i = i + 1

    return mias

def mia_best_th(model, train_data, test_data, device, nt=150):
    def search_th(Etrain, Etest):
      thrs = np.linspace(min(min(Etrain.min(), Etest.min()), 0), max(max(Etrain.max(), Etest.max()), 1), 100)
      R = np.zeros_like(thrs)

      for i, th in enumerate(thrs):
          tp = (Etrain < th).sum()
          tn = (Etest >= th).sum()
          acc = (tp + tn) / (len(Etrain) + len(Etest))  # Correcting the calculation for accuracy
          R[i] = acc

      return R.max()

    model.eval()
    Ltrain, Ptrain, Ytrain, null = evaluate(model, device, train_data)
    Ltest, Ptest, Ytest, Generalization_error = evaluate(model, device, test_data)
    model.train()

    thrs = ths_searching_space(nt, Ltrain, Ltest)
    loss_mia = search_th(Ltrain, Ltest)

    Etrain = compute_modified_entropy(Ptrain, Ytrain)
    Etest = compute_modified_entropy(Ptest, Ytest)

    thrs = ths_searching_space(nt, Etrain, Etest)
    ent_mia = search_th(Etrain, Etest)

    #print(f" Loss: {loss_mia}, Enthropy: {ent_mia}")

    return loss_mia, ent_mia, Generalization_error

def compute_modified_entropy(p, y, epsilon=0.00001):
    """ Computes label informed entropy from 'Systematic evaluation of privacy risks of machine learning models' USENIX21 """
    assert len(y) == len(p)
    n = len(p)

    entropy = np.zeros(n)

    for i in range(n):
        pi = p[i]
        yi = y[i]
        for j, pij in enumerate(pi):
            if j == yi:
                # right class
                entropy[i] -= (1-pij)*np.log(pij+epsilon)
            else:
                entropy[i] -= (pij)*np.log(1-pij+epsilon)

    return entropy


def ths_searching_space(nt, train, test):
    thrs = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), nt)
    return thrs

def evaluate(model, device, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x, y = data
    x, y = x.to(device), y.to(device)

    losses = []
    preds = []
    labels = []

    correct = 0
    total = 0

    for idx in range(len(x)):
        with torch.no_grad():
            scores = model(x[idx].unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(scores, y[idx].unsqueeze(0))

            # Collect probability scores instead of class predictions
            prob_scores = torch.nn.functional.softmax(scores, dim=-1).cpu().numpy()
            label = y[idx].cpu().numpy()

            losses.append(loss.cpu().numpy())
            preds.append(prob_scores.reshape(1, -1))  # Store probability scores
            labels.append(label.reshape(1, -1))  # Ensure labels are added as arrays

            predicted = np.argmax(prob_scores, axis=1)
            correct += (predicted == label).sum().item()
            total += label.size

    losses = np.array(losses)
    preds = np.concatenate(preds) if preds else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])

    accuracy = correct / total if total > 0 else 0.0
    generalization_error = 100.0 - accuracy

    return losses, preds, labels, generalization_error

def compute_consensus_distance(nodes) -> float:
    num_nodes = len(nodes)
    consensus_distance = 0.0

    for node_v in nodes.values():
        local_params_v = node_v.model_handler.model.state_dict()
        pairwise_distance = 0.0

        for node_u in nodes.values():
            if node_u.idx != node_v.idx:  # Exclude self-comparison
                local_params_u = node_u.model_handler.model.state_dict()
                distance = sum((local_params_v[key] - local_params_u[key]).pow(2).sum().item() for key in local_params_v)
                pairwise_distance += distance / (num_nodes ** 2 - num_nodes)

        consensus_distance += pairwise_distance

    return consensus_distance

def assign_model_params(source_model, target_model):
    device = next(target_model.parameters()).device
    source_state_dict = source_model.state_dict()
    target_model.load_state_dict(source_state_dict, strict=False)
    target_model.to(device)

def plot_mia_vulnerability(mia_accuracy, gen_error):
        plt.figure(figsize=(8, 6))
        plt.scatter(gen_error, mia_accuracy, label='MIA Vulnerability vs Generalization Error', color='blue')
        plt.xlabel('Generalization Error')
        plt.ylabel('MIA Vulnerability')
        plt.title('MIA Vulnerability over Generalization Error')
        plt.legend()
        plt.grid(True)
        plt.show()
