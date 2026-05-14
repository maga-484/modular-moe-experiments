import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset_poly import ModularPolynomialDataset
from modular_experiment.model import ModularMoE

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output, gate_weights = model(x)
        loss = criterion(output.squeeze(), y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / total_batches

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches

def train(modulo=5, num_experts=4, epochs=200, lr=0.001, hidden_dim=64):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = ModularPolynomialDataset(max_num=modulo-1, modulo=modulo, size=10000)
    val_dataset = ModularPolynomialDataset(max_num=modulo-1, modulo=modulo, size=2000, seed=43)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ModularMoE(input_dim=2, output_dim=1, num_experts=num_experts, hidden_dim=hidden_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    save_name = f"model_poly_mod{modulo}_experts_{num_experts}.pt"
    
    print(f"Entrenando polinomio (a²+b) con modulo={modulo}, num_experts={num_experts}")
    print(f"Device: {device}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'modulo': modulo,
                'num_experts': num_experts,
                'operation': 'polynomial'
            }, f"models_shared/{save_name}")
    
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modulo', type=int, default=5)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    os.makedirs("models_shared", exist_ok=True)
    
    model = train(modulo=args.modulo, num_experts=args.num_experts, epochs=args.epochs)