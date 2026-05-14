import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset_mul import AllCombinationsMulDataset
from modular_experiment.model import ModularMoE

def load_model(model_path, num_experts=4, hidden_dim=64):
    """Cargar modelo guardado"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModularMoE(input_dim=2, output_dim=1, num_experts=num_experts, 
                       hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('modulo', None)

def analyze_routing_mul(model, max_num=4, modulo=None, device='cpu'):
    """Analizar el patrón de enrutamiento para multiplicación"""
    dataset = AllCombinationsMulDataset(max_num=max_num, modulo=modulo)
    routing_matrix = np.zeros((max_num + 1, max_num + 1), dtype=int)
    gate_weight_matrix = np.zeros((max_num + 1, max_num + 1, model.num_experts))
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, (a, b) = dataset[i]
            x = x.unsqueeze(0).to(device)
            _, gate_weights = model(x)
            expert_idx = torch.argmax(gate_weights, dim=1).item()
            routing_matrix[a, b] = expert_idx
            gate_weight_matrix[a, b] = gate_weights.cpu().numpy().squeeze()
    
    return routing_matrix, gate_weight_matrix

def compute_entropy(routing_matrix):
    """Calcular entropía de la distribución de expertos"""
    unique, counts = np.unique(routing_matrix, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    return entropy

def compare_sum_vs_mul():
    """Comparar suma modular vs multiplicación modular"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelos de suma (resultados anteriores)
    sum_mod5_path = "models_shared/model_modulo_5_experts_4.pt"
    mul_mod5_path = "models_shared/model_mul_mod5_experts_4.pt"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Suma módulo 5
    if Path(sum_mod5_path).exists():
        model, _ = load_model(sum_mod5_path, num_experts=4)
        model = model.to(device)
        routing_sum, _ = analyze_routing_mul(model, max_num=4, modulo=5, device=device)
        
        ax = axes[0, 0]
        im = ax.imshow(routing_sum, cmap='tab10', vmin=0, vmax=3)
        ax.set_title(f'Suma Modular (mod 5)\nEntropía: 0.440', fontsize=12)
        ax.set_xlabel('b')
        ax.set_ylabel('a')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, routing_sum[i, j], ha='center', va='center', color='white')
        plt.colorbar(im, ax=ax)
    
    # Multiplicación módulo 5
    if Path(mul_mod5_path).exists():
        model, modulo = load_model(mul_mod5_path, num_experts=4)
        model = model.to(device)
        routing_mul, _ = analyze_routing_mul(model, max_num=4, modulo=5, device=device)
        entropy_mul = compute_entropy(routing_mul)
        
        ax = axes[0, 1]
        im = ax.imshow(routing_mul, cmap='tab10', vmin=0, vmax=3)
        ax.set_title(f'Multiplicación Modular (mod 5)\nEntropía: {entropy_mul:.3f}', fontsize=12)
        ax.set_xlabel('b')
        ax.set_ylabel('a')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, routing_mul[i, j], ha='center', va='center', color='white')
        plt.colorbar(im, ax=ax)
        
        # Mostrar resultados
        print("\n=== MULTIPLICACIÓN MODULAR (mod 5) ===")
        print("Matriz de enrutamiento:")
        print(routing_mul)
        print(f"Entropía: {entropy_mul:.4f}")
        
        # Análisis adicional
        unique, counts = np.unique(routing_mul, return_counts=True)
        print(f"Expertos activos: {len(unique)} de 4")
        for exp, count in zip(unique, counts):
            print(f"  Experto {exp}: {count/25*100:.1f}%")
    
    plt.suptitle('Comparación: Suma Modular vs Multiplicación Modular (mod 5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/sum_vs_mul_comparison.png', dpi=150)
    plt.show()

def analyze_multiplication_patterns(routing_matrix, modulo=5):
    """Analizar patrones específicos de multiplicación modular"""
    
    print("\n=== ANÁLISIS DE PATRONES ===")
    
    # Identificar ceros
    zero_positions = []
    for i in range(modulo):
        for j in range(modulo):
            if (i * j) % modulo == 0:
                zero_positions.append((i, j))
    
    print(f"Posiciones donde producto = 0: {len(zero_positions)} casos")
    
    # Ver qué experto maneja los ceros
    zero_experts = [routing_matrix[i, j] for i, j in zero_positions]
    from collections import Counter
    counter = Counter(zero_experts)
    print(f"Expertos que manejan ceros: {dict(counter)}")
    
    # Identificar unos
    one_positions = []
    for i in range(modulo):
        for j in range(modulo):
            if (i * j) % modulo == 1:
                one_positions.append((i, j))
    
    if one_positions:
        one_experts = [routing_matrix[i, j] for i, j in one_positions]
        counter_one = Counter(one_experts)
        print(f"Expertos que manejan unos: {dict(counter_one)}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("🔬 Analizando multiplicación modular...")
    compare_sum_vs_mul()