
import gc
import torch

from gossipy.model.architecture import resnet20

def clear_cuda_cache():
    try:
        tensor = tensor.cuda()
    except Exception as e:
        tensor = torch.tensor([1.0]).cuda()
        print("An error occurred while clearing CUDA cache:", e)
    del tensor
    try:
         model = model.cuda()
    except Exception as e:
        model = resnet20().cuda() 
        print("An error occurred while clearing CUDA cache:", e)
    del model
    torch.cuda.empty_cache()
    gc.collect()