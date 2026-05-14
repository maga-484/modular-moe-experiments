import numpy as np
import torch
from torch.utils.data import Dataset

class ModularPolynomialDataset(Dataset):
    """Dataset para polinomio cuadrático: (a² + b) mod n"""
    
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
                y = a*a + b  # polinomio normal
            else:
                y = (a*a + b) % modulo
            
            self.data.append(([a, b], y))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_polynomial(modulo=5, epochs=200):
    """Entrenar para polinomio modular"""
    from modular_experiment.train import train
    
    # Reutilizar el entrenamiento de suma pero con dataset diferente
    # (implementación rápida)
    print(f"Entrenando polinomio (a²+b) mod {modulo}...")
    # ... código similar a train.py