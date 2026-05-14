import numpy as np
import torch
from torch.utils.data import Dataset

class ModularPowerDataset(Dataset):
    """Dataset para la tarea de potencia modular (a^b mod n)"""
    
    def __init__(self, max_num=4, modulo=None, size=10000, seed=42):
        np.random.seed(seed)
        self.max_num = max_num
        self.modulo = modulo
        self.size = size
        
        self.data = []
        for _ in range(size):
            a = np.random.randint(0, max_num + 1)
            b = np.random.randint(0, max_num + 1)
            
            if modulo is None:
                # Potencia normal (puede crecer muy rápido)
                y = a ** b
            else:
                # Potencia modular
                y = pow(a, b, modulo)  # Más eficiente que (a**b) % modulo
            
            self.data.append(([a, b], y))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AllCombinationsPowDataset(Dataset):
    """Dataset con todas las combinaciones posibles de a y b"""
    
    def __init__(self, max_num=4, modulo=None):
        self.max_num = max_num
        self.modulo = modulo
        self.combinations = []
        
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                if modulo is None:
                    y = a ** b
                else:
                    y = pow(a, b, modulo)
                self.combinations.append(([a, b], y))
    
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx):
        x, y = self.combinations[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), (idx // (self.max_num + 1), idx % (self.max_num + 1))