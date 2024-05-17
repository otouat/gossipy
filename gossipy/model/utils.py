
import gc
import torch
import subprocess as sp
import os
from gossipy.model.architecture import resnet20
import nvidia_smi



def clear_cuda_cache():
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

    torch.cuda.empty_cache()
    gc.collect()

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
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
