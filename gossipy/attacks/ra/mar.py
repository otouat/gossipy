import torch
from collections import OrderedDict

def isolate_victim(model_update_buffer, victim_id):
    """ Computes marginalized model  """
    
    # Received model updates for the round
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

    # accumulate others
    other = init_list_variables(victim[1])
    for _, model in others:
        other = agg_sum(other, model)
    other = agg_div(other, n)

    # Check for NaNs after aggregation
    if any(torch.isnan(param).any() for param in other.values()):
        print("NaN detected after aggregation of others")

    # remove global functionality
    victim_c = agg_sub(victim[1], other)
    # Check for NaNs after subtraction
    if any(torch.isnan(param).any() for param in victim_c.values()):
        print("NaN detected after subtraction")

    # scale back marginalized model
    victim_c = agg_div(victim_c, 1/n)
    # Check for NaNs after scaling
    if any(torch.isnan(param).any() for param in victim_c.values()):
        print("NaN detected after scaling")

    return victim_c

def init_list_variables(A):
    B = OrderedDict()
    for key in A:
        B[key] = torch.zeros_like(A[key], dtype=torch.float32)  # Use higher precision
    return B

def agg_sum(A, B):
    assert len(A) == len(B)
    C = OrderedDict()
    for key in A:
        C[key] = A[key].to(torch.float32) + B[key].to(torch.float32)  # Ensure higher precision
    return C

def agg_sub(A, B):
    assert len(A) == len(B)
    C = OrderedDict()
    for key in A:
        C[key] = A[key].to(torch.float32) - B[key].to(torch.float32)  # Ensure higher precision
    return C

def agg_div(A, alpha):
    C = OrderedDict()
    for key in A:
        C[key] = A[key].to(torch.float32) / alpha  # Ensure higher precision
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