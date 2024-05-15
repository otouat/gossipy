from collections import OrderedDict
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
from gossipy.ra.mar import *
import copy

def ra_for_each_nn(simulation, attackerNode, final_agg, marginalized : bool = False):
    gradient = attackerNode.gradient
    reconstructed = invert_fully_g(gradient["fc1.weight"], gradient["fc1.bias"])
    attackerNode.gradient =  OrderedDict()
    return reconstructed

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
    print("ok")
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
    
def invert_fully_g(gw, gb, i=None, epsilon=0.00001):
    b = 1. / (gb.detach().numpy()[np.newaxis,:] + epsilon)
    w = gw.detach().numpy().T

    if not i is None:
        x = b[:, i] * w[i, :]
    else:
        x = (np.matmul(b, w))
        print(b.shape, w.shape, x.shape)
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
    plt.savefig(f'images_created2/test.png', dpi=300, bbox_inches='tight')
    plt.close()
