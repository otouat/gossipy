from collections import OrderedDict
import torch
import numpy as np
import math

def isolate_victim(model_update_buffer, victim_id):
    """ Computes marginalized model  """
    
    # Received model updates for the round
    thetas = model_update_buffer.copy()
    n = len(thetas)
    for i, (num, _) in enumerate(thetas):
        if num == victim_id:
            victim = thetas.pop(i)
            break
    others = thetas

    # accumulate others
    other = init_list_variables(victim[1].model.state_dict())
    for _, model in others:
        other = agg_sum(other, model.model.state_dict())
    other = agg_div(other, n)

    # remove global functionality
    victim_c = agg_sub(victim[1].model.state_dict(), other)
    # scale back marginalized model
    victim_c = agg_div(victim_c, 1/n)
    return victim_c

def assign_list_variables(A, B):
    state_dict = A
    B = []
    for key in state_dict:
        tensor_value = state_dict[key]
        B.append(tensor_value)
    return B

def init_list_variables(A):
    n = len(A)
    B = OrderedDict()
    for key in A:
        B[key] = 0
    return B

def agg_sum(A, B):
    assert(len(A) == len(B))
    n = len(A) 
    C = OrderedDict()
    for key in A:
        C[key] = A[key] + B[key]
    return C

def agg_sub(A, B):
    assert(len(A) == len(B))
    n = len(A) 
    C = OrderedDict()
    for key in A:
        C[key] = A[key] - B[key]
        
    return C

def agg_div(A, alpha):
    n = len(A) 
    C = OrderedDict()
    for key in A:
        C[key] = A[key] / alpha
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