
import gc
import torch


def clear_cuda_cache():
    tensor = tensor.cuda()  # Move tensor to GPU
    del tensor
    model = model.cuda()  # Move model to GPU
    # Perform operations with the model
    del model  # Delete model to free up GPU memory
    torch.cuda.empty_cache()
    gc.collect()
