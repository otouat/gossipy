from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
from typing import Tuple, Dict, List
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
import os
from gossipy.attacks.ra.mar import *
from torchsummary import summary


def mia_for_each_nn(simulation, attackerNode):
    class_specific = attackerNode.class_specific
    marginalized = attackerNode.marginalized_state
    nn = sorted(attackerNode.p2p_net.get_peers(attackerNode.idx), key=lambda x: int(x))
    
    mia_results = [[], []]
    # Ensure consistent structure for mia_results
    
    for node in simulation.nodes.values():
        if node.idx in nn:
            data = node.data
            train_data, test_data = data
            device = node.model_handler.device
            
            model = copy.deepcopy(node.model_handler.model)
            if marginalized:
                marginalized_state = isolate_victim(attackerNode.received_models, node.idx)
                model.load_state_dict(marginalized_state, strict=False)
                model.to(device)
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
            acc = (tp + tn) / (len(train) + len(test))  # Correcting the calculation for accuracy
            if log and False:
                print(f"Acc: {acc}")
            R[i] = acc

        return R.max()

    model.eval()
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data, log = log)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data, log = log)
    if log and True:
        print(f"Train size: {len(train_data)}")
        print("Mean loss train: ", np.mean(Ltrain))
        print("Mean loss test: ", np.mean(Ltest))
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
        
    if log:
        print(f"Loss_mia: {loss_mia}, Ent_mia: {ent_mia}")
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
    print("-2")
    Ltrain, Ptrain, Ytrain = evaluate(model, device, train_data)
    Ltest, Ptest, Ytest = evaluate(model, device, test_data)
    print("-1")
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

def add_random_black_box(image):
    """Adds a random black box to an image tensor."""
    h, w = image.shape[-2:]
    box_h, box_w = 10, 10  # Size of the black box
    x = np.random.randint(0, h - box_h)
    y = np.random.randint(0, w - box_w)
    image[:, x:x+box_h, y:y+box_w] = 0.0  # Set pixels in the black box to 0
    return image

def add_noise(image, mean=0, std=0.1):
    """Adds Gaussian noise to an image tensor."""
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, min=0.0, max=1.0)  # Ensure values are in [0, 1] range
    return noisy_image

def evaluate(model, device, data: Tuple[torch.Tensor, torch.Tensor], box=True, noise=True, log=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = data
    model = model.to(device)
    x, y = x.to(device), y.to(device)

    losses = []
    preds = []
    labels = []
    print("0")
    for idx in range(len(x)):
        print("1")
        input_tensor = x[idx].unsqueeze(0)  # Ensure the input tensor has shape [1, channels, height, width]
        original = input_tensor.clone()
        
        if noise:
            print("2")
            input_tensor = add_noise(input_tensor).to(device)  # Add Gaussian noise
        if box:
            print("3")
            input_tensor = add_random_black_box(input_tensor).to(device)  # Add random black box
        
        with torch.autocast(device_type="cuda"):
            print("4")
            scores = model(input_tensor)
            loss = torch.nn.functional.cross_entropy(scores, y[idx].unsqueeze(0))

            prob_scores = torch.nn.functional.softmax(scores, dim=-1).cpu().numpy()
            label = y[idx].cpu().numpy()

            losses.append(loss.cpu().numpy())
            preds.append(prob_scores.reshape(1, -1))  # Store probability scores
            labels.append(label.reshape(1, -1))  # Ensure labels are added as arrays
            
            # Inside evaluate function, before the model evaluation
            print(f"Noise applied: {noise}, Black box applied: {box}")
            # Inside evaluate function, before visualization
            print(f"Original tensor range: {original.min().item()} - {original.max().item()}")
            print(f"Modified tensor range: {input_tensor.min().item()} - {input_tensor.max().item()}")

            # After the tensor manipulation, directly visualize before model evaluation
            visualize_images(original, input_tensor.clone().cpu())
        print("5")
    losses = np.array(losses)
    preds = np.concatenate(preds) if preds else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    model = model.to("cpu")
    print("6")
    
    return losses, preds, labels

def visualize_images(original_images, modified_images):
    num_images = original_images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 4*num_images))
    
    for i in range(num_images):
        # Original image
        original_img = original_images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # Ensure [0, 1] range
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Modified image (noisy or with black box)
        modified_img = modified_images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # Ensure [0, 1] range
        axes[i, 1].imshow(modified_img)
        axes[i, 1].set_title('Modified Image')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()


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

