import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import copy

# Enhanced NaN check function
def check_for_nans(state_dict, label):
    for key, value in state_dict.items():
        if torch.isnan(value).any():
            print(f"NaN detected in {label} at {key}")

# Enhanced isolate_victim function with NaN checks
def isolate_victim(model_update_buffer, victim_id):
    """ Computes marginalized model  """
    
    thetas = model_update_buffer.copy()
    n = len(thetas)
    victim = None
    for i, (num, _) in enumerate(thetas):
        if num == victim_id:
            victim = thetas.pop(i)
            break
    if victim is None:
        raise ValueError(f"Victim ID {victim_id} not found in the model update buffer.")
    
    others = thetas

    other = init_list_variables(victim[1])
    for _, model in others:
        other = agg_sum(other, model)
    other = agg_div(other, n)

    check_for_nans(other, "aggregation of others")

    victim_c = agg_sub(victim[1], other)
    check_for_nans(victim_c, "subtraction")

    victim_c = agg_div(victim_c, 1/n)
    check_for_nans(victim_c, "scaling")

    return victim_c


def init_list_variables(A):
    B = OrderedDict()
    for key in A:
        if 'bn' not in key:  # Skip BatchNorm layers
            B[key] = torch.zeros_like(A[key], dtype=torch.float32)
        else:
            B[key] = A[key].clone()  # Just clone the BatchNorm layer parameters
    return B

def agg_sum(A, B):
    assert len(A) == len(B)
    C = OrderedDict()
    for key in A:
        if 'bn' not in key:  # Skip BatchNorm layers
            C[key] = A[key].to(torch.float32) + B[key].to(torch.float32)
        else:
            C[key] = A[key]  # Keep BatchNorm parameters unchanged
    return C

def agg_sub(A, B):
    assert len(A) == len(B)
    C = OrderedDict()
    for key in A:
        if 'bn' not in key:  # Skip BatchNorm layers
            C[key] = A[key].to(torch.float32) - B[key].to(torch.float32)
        else:
            C[key] = A[key]  # Keep BatchNorm parameters unchanged
    return C

def agg_div(A, alpha):
    C = OrderedDict()
    for key in A:
        if 'bn' not in key:  # Skip BatchNorm layers
            C[key] = A[key].to(torch.float32) / alpha
        else:
            C[key] = A[key]  # Keep BatchNorm parameters unchanged
    return C

def agg_neg(A):
    n = len(A) 
    C = OrderedDict()
    for key in A:
        C[key] = -A[key]
    return C

def agg_sumc(A, B):
    n = len(A) 
    C = OrderedDict()
    for key in A:
        C[key] = A[key] + B[key]
    return C