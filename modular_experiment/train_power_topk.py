import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset_pow import ModularPowerDataset
from modular_experiment.model_topk import TopKMoE
from modular_experiment.train_topk import train_epoch, evaluate

def train_power_topk(modulo=5, num_experts=4, top_k=2, epochs=200,
                     load_balance_weight=0.05, lr=0.001, hidden_dim=64):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets
    train_dataset = ModularPowerDataset(max_num=modulo-1, modulo=modulo, size=10000)
    val_dataset = ModularPowerDataset(max_num=modulo-1, modulo=modulo, size=2000, seed=43)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Modelo Top-k
    model = TopKMoE(input_dim=2, output_dim=1, num_experts=num_experts,
                    top_k=top_k, hidden_dim=hidden_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    save_name = f"topk_pow_mod{modulo}_exp{num_experts}_k{top_k}.pt"
    
    print(f"Entrenando Top-{top_k} MoE para POTENCIA modulo={modulo}, num_experts={num_experts}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, load_loss = train_epoch(model, train_loader, optimizer, criterion,
                                            device, load_balance_weight)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, Load Loss = {load_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'modulo': modulo,
                'num_experts': num_experts,
                'top_k': top_k,
                'operation': 'power'
            }, f"models_shared/{save_name}")
    
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modulo', type=int, default=5)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    os.makedirs("models_shared", exist_ok=True)
    
    model = train_power_topk(modulo=args.modulo, num_experts=args.num_experts,
                             top_k=args.top_k, epochs=args.epochs)