from typing import Tuple
import numpy as np
from sklearn.utils import resample
import torch
import torch.nn.functional as F
import copy

from gossipy.attacks.ra.mar import isolate_victim

def mia_for_each_nn(simulation, attackerNode):
    class_specific = attackerNode.class_specific
    marginalized = attackerNode.marginalized_state
    nn = sorted(attackerNode.p2p_net.get_peers(attackerNode.idx), key=lambda x: int(x))
    
    if class_specific:
        mia_results = [[], []]
    else:
        mia_results = [[], []]  # Ensure consistent structure for mia_results
    
    for node in simulation.nodes.values():
        if node.idx in nn:
            data = node.data
            train_data, test_data = data
            device = node.model_handler.device
            
            model = copy.deepcopy(node.model_handler.model)
            if marginalized:
                print("Marginalized mia")
                marginalized_state = isolate_victim(attackerNode.received_models, node.idx)
                model.load_state_dict(marginalized_state, strict=False)
                model.to(device)
                print("Marginalized model loaded")
                check_for_nans_in_model(model)
                print("Marginalized model checked")
                loss_mia, ent_mia = mia_best_th(model, train_data, test_data, device, log=False)
                mia_results[0].append(loss_mia)
                mia_results[1].append(ent_mia)
            elif class_specific:
                num_classes = max(train_data[1].max().item(), test_data[1].max().item()) + 1
                results = mia_best_th_class(model, train_data, test_data, num_classes, device)
                mia_results[0].append(results[0])
                mia_results[1].append(results[1])
            else:
                loss_mia, ent_mia = mia_best_th(model, train_data, test_data, device)
                mia_results[0].append(loss_mia)
                mia_results[1].append(ent_mia)
    
    if len(mia_results[0]) == 0 or len(mia_results[1]) == 0:
        raise ValueError("mia_results does not have enough elements")
    
    mia_results = {
        "loss_mia": np.mean(mia_results[0]),
        "entropy_mia": np.mean(mia_results[1])
    }
    return mia_results

def check_for_nans_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf detected in {name}")

def mia_best_th(model, train_data, test_data, device, nt=200, log=False):
    def search_th(train, test):
        thrs = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), nt)
        R = np.zeros_like(thrs)

        for i, th in enumerate(thrs):
            tp = (train < th).sum()
            tn = (test >= th).sum()
            acc = (tp + tn) / (len(train) + len(test))
            if log:
                print(f"Acc: {acc}")
            R[i] = acc

        return R.max()

    model.eval()
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data, log=log)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data, log=log)
    if log:
        print(f"Train size: {len(train_data)}")
        print("Mean loss train: ", np.mean(Ltrain))
        print("Mean loss test: ", np.mean(Ltest))
    model.train()

    n = min(Ptest.shape[0], Ptrain.shape[0])
    Ptrain = Ptrain[:n]
    Ytrain = Ytrain[:n]
    Ltrain = Ltrain[:n]
    loss_mia = search_th(Ltrain, Ltest)
    
    Etrain = compute_modified_entropy(Ptrain, Ytrain)
    Etest = compute_modified_entropy(Ptest, Ytest)
    ent_mia = search_th(Etrain, Etest)
        
    if log:
        print(f"Loss_mia: {loss_mia}, Ent_mia: {ent_mia}")

    return loss_mia, ent_mia

def mia_best_th_class(model, train_data, test_data, num_class, device, nt=200):
    def search_th(train, test, train_label, test_label, num_classes):
        thrs = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), nt)
        R = np.zeros((num_classes, len(thrs)))
        train_c = []
        test_c = []

        for c in range(num_classes):
            for i in range(min(len(train), len(test))):
                if train_label[i] == c:
                    train_c.append(train[i])
                if test_label[i] == c:
                    test_c.append(test[i])
            train_c = resample(train_c, replace=False, n_samples=min(int(sum(train_label == c)), int(sum(test_label == c))))
            test_c = resample(test_c, replace=False, n_samples=min(int(sum(train_label == c)), int(sum(test_label == c))))

            for i, th in enumerate(thrs):
                tp = (train_c < th).sum()
                tn = (test_c >= th).sum()
                acc = (tp + tn) / (len(train_c) + len(test_c))

                R[c, i] = acc

        return R.max(axis=1)

    model.eval()
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data)
    model.train()

    n = min(Ptest.shape[0], Ptrain.shape[0])
    Ptrain = Ptrain[:n]
    Ytrain = Ytrain[:n]
    Ltrain = Ltrain[:n]

    th_indices_loss = search_th(Ltrain, Ltest, Ytrain, Ytest, num_class)
    th_indices_ent = search_th(compute_modified_entropy(Ptrain, Ytrain), compute_modified_entropy(Ptest, Ytest), Ytrain, Ytest, num_class)

    return th_indices_loss, th_indices_ent

def compute_modified_entropy(p, y, epsilon=0.00001):
    assert len(y) == len(p)
    n = len(p)

    entropy = np.zeros(n)

    for i in range(n):
        pi = p[i]
        yi = y[i]
        for j, pij in enumerate(pi):
            if j == yi:
                entropy[i] -= (1-pij) * np.log(pij + epsilon)
            else:
                entropy[i] -= pij * np.log(1-pij + epsilon)

    return entropy

def evaluate(model, device, data: Tuple[torch.Tensor, torch.Tensor], log=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = data
    model = model.to(device)
    x, y = x.to(device), y.to(device)

    losses = []
    preds = []
    labels = []

    for idx in range(len(x)):
        input_tensor = x[idx].unsqueeze(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                scores = model(input_tensor)
                loss = torch.nn.functional.cross_entropy(scores, y[idx].unsqueeze(0))
                if torch.isnan(scores).any():
                    print("NaN detected in scores")
                if torch.isnan(loss).any():
                    print("NaN detected in loss")
                prob_scores = torch.nn.functional.softmax(scores, dim=-1).cpu().numpy()
                label = y[idx].cpu().numpy()

                losses.append(loss.cpu().numpy())
                preds.append(prob_scores.reshape(1, -1))
                labels.append(label.reshape(1, -1))
    
    losses = np.array(losses)
    preds = np.concatenate(preds) if preds else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    model = model.to("cpu")
    return losses, preds, labels


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

def inspect_model(model):
    # Print model architecture
    print(model)
    # Print model parameters
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Size: {param.size()}")

def check_model_initialization(original_model, marginalized_model):
    # Print state dictionaries of original and marginalized models
    print("Original Model State Dictionary:")
    #print(original_model.state_dict())
    print("Marginalized Model State Dictionary:")
    #print(marginalized_model.state_dict())
    
    # Compare model architectures
    if original_model.__class__ != marginalized_model.__class__:
        print("Error: Model classes are different.")
    elif original_model.state_dict().keys() != marginalized_model.state_dict().keys():
        print("Error: Model state dictionaries do not match.")
    else:
        print("Model initialization successful. State dictionaries match.")
