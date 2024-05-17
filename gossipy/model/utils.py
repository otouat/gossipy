
import gc
import torch
import subprocess as sp
import os
from gossipy.model.architecture import resnet20
import nvidia_smi



def clear_cuda_cache():
    print("1:")
    get_gpu_memory()
    #get_nvdia_memory()
    log_remaining_tensors()
    
    try:
        # Initialize a tensor to ensure it can be moved to CUDA
        tensor = torch.tensor([1.0]).cuda()
    except Exception as e:
        print("An error occurred while clearing CUDA cache:", e)
    finally:
        del tensor

    try:
        # Initialize a model to ensure it can be moved to CUDA
        model = resnet20(10).cuda()
    except Exception as e:
        print("An error occurred while clearing CUDA cache:", e)
    finally:
        del model
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    print("2:")
    get_gpu_memory()
    #get_nvdia_memory()
    log_remaining_tensors()
    
def clear_memory():
    # Manually delete references to tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    
    # Collect garbage
    gc.collect()
    
    # Empty the CUDA cache
    torch.cuda.empty_cache()

def clear_cache_and_retry(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory: {e}")
        print("Clearing cache and retrying...")
        torch.cuda.empty_cache()
        clear_memory()
        func(*args, **kwargs)

def get_nvdia_memory():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    nvidia_smi.nvmlShutdown()

def get_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Max Cached: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")
    print()

def log_remaining_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(f"Remaining Tensor: {type(obj)}, Size: {obj.size()}")
                del obj
        except:
            pass