import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.model_topk import TopKMoE
from modular_experiment.model import ModularMoE
from modular_experiment.train_topk import evaluate
from modular_experiment.dataset import ModularAdditionDataset
from torch.utils.data import DataLoader

def compare_architectures(modulo=5):
    """Comparar Top-k MoE vs Full MoE (todos los expertos)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar Full MoE (entrenado previamente)
    full_model_path = f"models_shared/model_modulo_{modulo}_experts_4.pt"
    topk_model_path = f"models_shared/topk_mod{modulo}_exp4_k2.pt"
    
    results = {}
    
    # Full MoE
    if Path(full_model_path).exists():
        full_model = ModularMoE(input_dim=2, output_dim=1, num_experts=4)
        checkpoint = torch.load(full_model_path, map_location=device)
        full_model.load_state_dict(checkpoint['model_state_dict'])
        full_model.to(device)
        full_model.eval()
        results['Full MoE'] = {'model': full_model, 'checkpoint': checkpoint}
    else:
        print(f"⚠️ Modelo Full MoE no encontrado: {full_model_path}")
    
    # Top-k MoE
    if Path(topk_model_path).exists():
        topk_model = TopKMoE(input_dim=2, output_dim=1, num_experts=4, top_k=2)
        checkpoint = torch.load(topk_model_path, map_location=device)
        topk_model.load_state_dict(checkpoint['model_state_dict'])
        topk_model.to(device)
        topk_model.eval()
        results['Top-k MoE (k=2)'] = {'model': topk_model, 'checkpoint': checkpoint}
    else:
        print(f"⚠️ Modelo Top-k no encontrado. Ejecuta primero:")
        print(f"  python -m modular_experiment.train_topk --modulo {modulo}")
    
    # Dataset de prueba
    val_dataset = ModularAdditionDataset(max_num=modulo-1, modulo=modulo, size=2000, seed=999)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = torch.nn.MSELoss()
    
    print(f"\n{'='*50}")
    print(f"COMPARATIVA PARA MÓDULO {modulo}")
    print(f"{'='*50}")
    
    for name, data in results.items():
        model = data['model']
        checkpoint = data['checkpoint']
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"\n{name}:")
        print(f"  Mejor Val Loss: {checkpoint['val_loss']:.6f}")
        print(f"  Val Loss final: {val_loss:.6f}")
        
        if 'top_k' in checkpoint:
            print(f"  Top-k: {checkpoint['top_k']}")
        print(f"  Época: {checkpoint['epoch']}")
    
    return results

def compare_all():
    """Comparar para todos los módulos"""
    modulos = [5, 6, 7, 10]
    
    print("\n" + "="*60)
    print("COMPARATIVA COMPLETA: Full MoE vs Top-k MoE")
    print("="*60)
    
    for modulo in modulos:
        compare_architectures(modulo)

if __name__ == "__main__":
    compare_all()