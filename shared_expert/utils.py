import psutil
import torch

def get_memory_usage():
    ram = psutil.Process().memory_info().rss / 1024**2
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2
    return ram, vram
