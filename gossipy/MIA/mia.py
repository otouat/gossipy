from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from gossipy import LOG
from typing import List, Dict

def mia_for_each_nn(simulation, class_specific: bool = False):
    idx = simulation.attackerNode.idx
    nn = sorted(simulation.attackerNode.p2p_net.get_peers(idx), key=lambda x: int(x))
    model = copy.deepcopy(simulation.attackerNode.model_handler.model)
    mia_results = [[], []] if class_specific else []
    for node in simulation.nodes.values():
        if node.idx in nn:
            data = node.data
            train_data, test_data = data
            train_data = node.model_handler.get_trained_data()
            device = node.model_handler.device
            if class_specific:
                num_classes = max(train_data[1].max().item(), test_data[1].max().item())+1
                print(num_classes)
                results= mia_best_th_class(model, train_data, test_data, num_classes, device)
                mia_results[0].append(results[0])
                mia_results[1].append(results[1])
                #for class_idx, (loss_mia, ent_mia) in mia_results.items():
                    #print(f"Class {class_idx} Loss MIA: {np.mean(loss_mia)}")
                    #print(f"Class {class_idx} Entropy MIA: {np.mean(ent_mia)}")

            else:
                mia_results.append(mia_best_th(model, train_data, test_data, device))

    print("-----------------------------")
    print("Round MIA Results:")
    print(f"Mean Loss MIA: {np.mean(mia_results[0])}")
    print(f"Mean Entropy MIA: {np.mean(mia_results[1])}")

    return mia_results

def mia_best_th(model, train_data, test_data, device, nt=150):
    
    def search_th(train, test):
        thrs = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), nt)
        R = np.zeros_like(thrs)

        for i, th in enumerate(thrs):
            tp = (train < th).sum()
            tn = (test >= th).sum()
            acc = (tp + tn) / (len(train) + len(test))  # Correcting the calculation for accuracy
            R[i] = acc

        return R.max()

    model.eval()
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data)
    model.train()

    # it takes a subset of results on test set with size equal to the one of the training test 
    n = min(Ptest.shape[0], Ptrain.shape[0])
    Ptrain = Ptrain[:n]
    Ytrain = Ytrain[:n]
    Ltrain = Ltrain[:n]

    loss_mia = search_th(Ltrain, Ltest)

    Etrain = compute_modified_entropy(Ptrain, Ytrain)
    Etest = compute_modified_entropy(Ptest, Ytest)

    ent_mia = search_th(Etrain, Etest)

    #print(f" Loss: {loss_mia}, Enthropy: {ent_mia}")

    return loss_mia, ent_mia

def mia_best_th_class(model, train_data, test_data, num_class, device, nt=200):
    
    def search_th(train, test, train_label, test_label, num_classes):
        thrs = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), nt)
        R = np.zeros((num_classes, len(thrs)))  # Initialize array for class-specific thresholds
        train_c = []
        test_c = []

        for c in range(num_classes):
            for i in range(min(len(train), len(test))):
                if train_label[i] == c:
                    train_c.append(train[i])
                if test_label[i] == c:
                    test_c.append(test[i])
            # Resample the training and testing data to have an equal size in each class
            train_c = resample(train_c, replace=False, n_samples=min(int(sum(train_label == c)), int(sum(test_label == c))))
            test_c = resample(test_c, replace=False, n_samples=min(int(sum(train_label == c)), int(sum(test_label == c))))

            for i, th in enumerate(thrs):
                # Count true positives and true negatives for each class
                tp = (train_c < th).sum()
                tn = (test_c >= th).sum()
                acc = (tp + tn) / (len(train_c) + len(test_c))

                R[c, i] = acc

        return R.max(axis=1)

    model.eval()
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data)
    model.train()

    # it takes a subset of results on test set with size equal to the one of the training test
    n = min(Ptest.shape[0], Ptrain.shape[0])
    
    Ptrain = Ptrain[:n]
    Ytrain = Ytrain[:n]
    Ltrain = Ltrain[:n]

    th_indices_loss = search_th(Ltrain, Ltest, Ytrain, Ytest, num_class,)  # Class-specific thresholds for loss
    th_indices_ent = search_th(compute_modified_entropy(Ptrain, Ytrain), compute_modified_entropy(Ptest, Ytest), Ytrain, Ytest, num_class,)  # Class-specific thresholds for entropy

    return th_indices_loss, th_indices_ent

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

def evaluate(model, device, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x, y = data
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = x.to(device), y.to(device)

    losses = []
    preds = []
    labels = []

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

def compute_gen_errors(Simul, nodes) -> float:
    aggregated_acc_train = []
    aggregated_acc_test = []
    gen_error = []

    for _, node in nodes.items():
        acc_train = node.evaluate(node.data[0])["accuracy"]
        aggregated_acc_train.append(acc_train)
        if node.has_test():
            if Simul.data_dispatcher.has_test():
                acc_test = node.evaluate(node.data[1])["accuracy"]
                #acc_test = node.evaluate(self.data_dispatcher.get_eval_set())["accuracy"]
            else:
                acc_test = node.evaluate(node.data[1])["accuracy"]
            aggregated_acc_test.append(acc_test)

    # Compute the generalization error based on aggregated accuracies
    if aggregated_acc_train and aggregated_acc_test:
        avg_acc_train = sum(aggregated_acc_train) / len(aggregated_acc_train)
        avg_acc_test = sum(aggregated_acc_test) / len(aggregated_acc_test)
        gen_error_value = (avg_acc_train - avg_acc_test) / (avg_acc_test + avg_acc_train)
        gen_error.append(gen_error_value)
    else:
        gen_error.append(0)

    return gen_error
def get_gen_errors(acc_train, acc_test) -> float:
    return (acc_train - acc_test) / (acc_test + acc_train)

def assign_model_params(source_model, target_model):
    device = next(target_model.parameters()).device
    source_state_dict = source_model.state_dict()
    target_model.load_state_dict(source_state_dict, strict=False)
    target_model.to(device)

def plot_mia_vulnerability(mia_accuracy, gen_error):
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(gen_error, mia_accuracy, label='MIA Vulnerability over Generalization Error', color='blue')
        plt.xlabel('Generalization Error')
        plt.ylabel('MIA Vulnerability')
        plt.title('MIA Vulnerability over Generalization Error')
        plt.legend()
        plt.grid(True)
        plt.show()
        fig2 = plt.figure(figsize=(8, 6))
        epoch = list(range(1, len(mia_accuracy) + 1))
        plt.plot(epoch, mia_accuracy, label='MIA Vulnerability per epoch', color='green')
        plt.xlabel('Epoch n°')
        plt.ylabel('MIA Vulnerability')
        plt.title('MIA Vulnerability per epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
        return fig, fig2

import os
import matplotlib.pyplot as plt
import json
from datetime import datetime

def log_results(Simul, n_rounds, diagrams, global_evaluations):
    base_folder_path = os.path.join(os.getcwd(), "results")
    exp_tracker_file = os.path.join(base_folder_path, "exp_number.txt")

    # Read the last experiment number and increment it
    if os.path.exists(exp_tracker_file):
        with open(exp_tracker_file, 'r') as file:
            experiment_number = int(file.read().strip()) + 1
    else:
        experiment_number = 1

    # Create new subfolder
    new_folder_path = f"{base_folder_path}/Exp n°{experiment_number}"
    os.makedirs(new_folder_path, exist_ok=True)

    # Log file path for experiment parameters
    params_file_path = f"{new_folder_path}/simulation_params.log"

    # Log experiment parameters
    with open(params_file_path, 'w') as params_file:
        params_file.write(f"Experiment Number: {experiment_number}\n")
        params_file.write(f"Protocol: {type(Simul).__name__}\n")
        params_file.write(f"Timestamp: {datetime.now()}\n")
        params_file.write(f"Total Nodes: {Simul.n_nodes}\n")
        params_file.write(f"Total Rounds: {n_rounds}\n")

    # Log file path for simulation results
    log_file_path = f"{new_folder_path}/simulation_results.log"

    # Log simulation results
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Experiment Number: {experiment_number}\n")
        log_file.write(f"Timestamp: {datetime.now()}\n")

        log_file.write("\nMIA Evaluations:\n")
        log_file.write(f"MIA Vulnerability: {Simul.mia_accuracy}\n")
        log_file.write(f"Gen Error: {Simul.gen_error}\n")

        log_file.write("\nGlobal Evaluations:\n")
        for round_number, evaluation_dict in global_evaluations:
            log_file.write(f"Round {round_number}:\n")
            for key, value in evaluation_dict.items():
                log_file.write(f"{key}: {value}\n")

    # Save diagrams
    for name, fig in diagrams.items():
        fig.savefig(f"{new_folder_path}/{name}.png")

    # Update the experiment number tracker file
    with open(exp_tracker_file, 'w') as file:
        file.write(str(experiment_number))

def get_fig_evaluation(evals: List[List[Dict]],
                    title: str="Untitled plot") -> None:

    if not evals or not evals[0] or not evals[0][0]: return
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(111)
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu: float = np.mean(evs, axis=0)
        std: float = np.std(evs, axis=0)
        plt.fill_between(range(1, len(mu)+1), mu-std, mu+std, alpha=0.2)
        plt.title(title)
        plt.xlabel("cycle")
        plt.ylabel("metric value")
        plt.plot(range(1, len(mu)+1), mu, label=k)
        LOG.info(f"{k}: {mu[-1]:.2f}")
    ax.legend(loc="lower right")
    plt.show()

    return fig