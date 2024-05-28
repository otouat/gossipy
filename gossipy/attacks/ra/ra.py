from collections import OrderedDict
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
from gossipy.attacks.ra.mar import *
import copy

def ra_for_each_nn(victim, marginalized : bool = False):
    gradient = victim.gradient
    if 'fc.weight' in gradient and 'fc.bias' in gradient:
        weight = gradient['fc.weight']
        bias = gradient['fc.bias']
        reconstructed = invert_fully_g(weight, bias)
        print_images(reconstructed)
        return reconstructed
    else:
        print("Error: Missing 'fc.weight' or 'fc.bias' in gradient")
        return None  # Or handle appropriately


def sum_nested_structures_and_negate(structures):
    # Vérifie que 'structures' est une liste (ou un iterable) contenant au moins un élément
    if not structures or not all(isinstance(s, OrderedDict) for s in structures):
        raise ValueError("Le paramètre 'structures' doit être une liste d'OrderedDicts avec des valeurs tensorielles")            
    # Initialisation du résultat avec une copie profonde du premier élément pour garder les clés
    result = OrderedDict()
    for key in structures[0]:
        result[key] = 0

    # Itération sur chaque structure pour effectuer la somme
    for structure in structures[:-1]:
        for key in structure:
            # Accumulation des sommes des tenseurs pour chaque clé
            result[key] += structure[key]            
    # Négation des résultats accumulés
    for key in result:
        result[key] = (result[key] * int(1 / (len(structures) - 2))).long()
    return result

def w_fully_adv_init(W, mean, std, s):
    n, L = W.shape
    
    r = np.arange(L)
    _W = W.detach().numpy().copy()
    
    for i in range(n):
        mask = np.zeros(L)
        ids = np.random.choice(r, size=L//2, replace=False)
        mask[ids] = 1
        mask = mask.astype(bool)

        N = r[mask]
        P = r[~mask]

        zn = np.random.normal(mean, std, size=(L//2))
        zp = (-s) * zn
        np.random.shuffle(zp)
        _W[i, N] = zn
        _W[i, P] = zp

    W.data = torch.from_numpy(_W)
    
def invert_fully_g(weight, bias):
    # Print shapes for debugging
    print(f"weight shape: {weight.shape}, bias shape: {bias.shape}")
    
    # Reshape bias if necessary
    if bias.ndim == 1:
        bias = bias.reshape(1, -1)  # Reshape to (1, 10)

    # Transpose weight to match the multiplication requirement
    weight = weight.T  # Transpose weight to shape (64, 10)
    
    try:
        x = np.matmul(bias, weight)  # Now shapes (1, 10) and (10, 64) match
    except ValueError as e:
        print(f"Error during matmul: {e}")
        return None
    
    return x


def normalize_img(x):
    x += x.min()
    x -= x.min()
    x /= x.max()
    return x

def select_nn_mus(keys, buff):
    """ get only a subset of a dictonary """
    new_buff = {}
    for key in keys:
        name = key.name
        new_buff[name] = buff[name]
    return new_buff

def modify_filter(model, layer, i, j):
    kernel = model[layer]
    _kernel = kernel.detach().numpy()
    p1 = math.floor(_kernel.shape[0]/2)
    p2 = math.floor(_kernel.shape[1]/2)
    _kernel[:,:,i,j] = 1
    kernel.data = torch.from_numpy(_kernel)

def print_images(reconstructed_image):

    ax_reconstructed = plt.subplot()
    ax_reconstructed.imshow(np.transpose(reconstructed_image, [1, 2, 0]))
    # ax_reconstructed.set_title("original image") 
    plt.savefig(f'images_created/test.png', dpi=300, bbox_inches='tight')
    plt.close()
