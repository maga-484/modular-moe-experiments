import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset import ModularAdditionDataset
from modular_experiment.model_topk import TopKMoE, TopKMoEAnalyzer

def train_epoch(model, dataloader, optimizer, criterion, device, 
                load_balance_weight=0.01):
    model.train()
    total_loss = 0
    total_load_loss = 0
    total_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output, gate_info = model(x, return_gate=True)
        
        # Pérdida principal
        main_loss = criterion(output.squeeze(), y)
        
        # Pérdida auxiliar de balance de carga
        load_loss = TopKMoEAnalyzer.compute_load_balancing_loss(
            gate_info['logits'], 
            gate_info['top_k_indices'], 
            gate_info['top_k_weights']
        )
        
        # Pérdida total
        loss = main_loss + load_balance_weight * load_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += main_loss.item()
        total_load_loss += load_loss.item()
        total_batches += 1
    
    return total_loss / total_batches, total_load_loss / total_batches

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches

def train(modulo=5, num_experts=4, top_k=2, epochs=200, batch_size=32,
          lr=0.001, hidden_dim=64, load_balance_weight=0.01, seed=42):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Datasets
    train_dataset = ModularAdditionDataset(max_num=modulo-1, modulo=modulo, size=10000, seed=seed)
    val_dataset = ModularAdditionDataset(max_num=modulo-1, modulo=modulo, size=2000, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modelo Top-k MoE
    model = TopKMoE(input_dim=2, output_dim=1, num_experts=num_experts,
                    top_k=top_k, hidden_dim=hidden_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    save_name = f"topk_mod{modulo}_exp{num_experts}_k{top_k}.pt"
    
    print(f"Entrenando Top-{top_k} MoE con modulo={modulo}, num_experts={num_experts}")
    print(f"Load balance weight: {load_balance_weight}")
    print(f"Device: {device}")
    
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
                'architecture': 'topk_moe'
            }, f"models_shared/{save_name}")
    
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modulo', type=int, default=5)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--load_balance_weight', type=float, default=0.01)
    
    args = parser.parse_args()
    os.makedirs("models_shared", exist_ok=True)
    
    model = train(modulo=args.modulo, num_experts=args.num_experts,
                  top_k=args.top_k, epochs=args.epochs,
                  load_balance_weight=args.load_balance_weight)