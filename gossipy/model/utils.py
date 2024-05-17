
import gc
import torch

from gossipy.model.architecture import resnet20

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
