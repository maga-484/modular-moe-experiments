import numpy as np
import torch
from torch.utils.data import Dataset

class ModularMultiplicationDataset(Dataset):
    """Dataset para la tarea de multiplicación modular"""
    
    def __init__(self, max_num=4, modulo=None, size=10000, seed=42):
        """
        Args:
            max_num: Número máximo para a y b (0 a max_num)
            modulo: Si es None, multiplicación normal; si es entero, multiplicación módulo ese valor
            size: Cantidad de ejemplos
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.max_num = max_num
        self.modulo = modulo
        self.size = size
        
        self.data = []
        for _ in range(size):
            a = np.random.randint(0, max_num + 1)
            b = np.random.randint(0, max_num + 1)
            
            if modulo is None:
                y = a * b  # multiplicación normal
            else:
                y = (a * b) % modulo
            
            self.data.append(([a, b], y))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AllCombinationsMulDataset(Dataset):
    """Dataset con todas las combinaciones posibles de a y b (para evaluación)"""
    
    def __init__(self, max_num=4, modulo=None):
        self.max_num = max_num
        self.modulo = modulo
        self.combinations = []
        
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                if modulo is None:
                    y = a * b
                else:
                    y = (a * b) % modulo
                self.combinations.append(([a, b], y))
    
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx):
        x, y = self.combinations[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), (idx // (self.max_num + 1), idx % (self.max_num + 1))