import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset import ModularAdditionDataset, AllCombinationsDataset
from modular_experiment.model import ModularMoE
from modular_experiment.train import train
from modular_experiment.eval_routing import analyze_routing, compute_expert_specialization

def run_experiment_mod10():
    """Experimento con módulo 10 (números 0-9)"""
    
    # Crear directorios
    os.makedirs("models_shared", exist_ok=True)
    os.makedirs("results/experiment_mod10", exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENTO 1: MÓDULO 10 (0-9)")
    print("=" * 60)
    
    # Entrenar modelo con módulo 10
    print("\nEntrenando modelo para módulo 10...")
    model = train(
        max_num=9,  # Números del 0 al 9
        modulo=10,
        num_experts=4,
        epochs=300,  # Más épocas para problema más complejo
        hidden_dim=128,  # Más capacidad
        batch_size=64
    )
    
    # Evaluar especialización
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("\nAnalizando patrones de enrutamiento...")
    routing_matrix, gate_weights = analyze_routing(
        model, max_num=9, modulo=10, device=device
    )
    
    print("\nMatriz de enrutamiento (10x10):")
    print(routing_matrix)
    
    specialization = compute_expert_specialization(
        model, max_num=9, modulo=10, device=device
    )
    
    print(f"\nDistribución de expertos: {specialization['expert_distribution']}")
    print(f"Entropía promedio del gate: {specialization['avg_entropy']:.4f}")
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Matriz de enrutamiento
    im1 = axes[0].imshow(routing_matrix, cmap='tab10', vmin=0, vmax=9)
    axes[0].set_title(f'Routing Decisions - Modulo 10\nEntropy: {specialization["avg_entropy"]:.3f}', 
                      fontsize=14)
    axes[0].set_xlabel('b', fontsize=12)
    axes[0].set_ylabel('a', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Expert ID')
    
    # Distribución de expertos
    experts = range(len(specialization['expert_distribution']))
    axes[1].bar(experts, specialization['expert_distribution'])
    axes[1].set_title('Expert Distribution', fontsize=14)
    axes[1].set_xlabel('Expert ID', fontsize=12)
    axes[1].set_ylabel('Fraction of Samples', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("results/experiment_mod10/routing_analysis.png", dpi=150)
    plt.show()
    
    # Análisis adicional: ver si hay especialización por suma
    analyze_by_sum_value(routing_matrix, max_num=9, modulo=10)
    
    return model, routing_matrix, specialization

def analyze_by_sum_value(routing_matrix, max_num=9, modulo=10):
    """Analizar qué experto se activa para cada valor de suma"""
    sum_expert_map = {}
    
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            sum_val = (a + b) % modulo
            expert = routing_matrix[a, b]
            
            if sum_val not in sum_expert_map:
                sum_expert_map[sum_val] = []
            sum_expert_map[sum_val].append(expert)
    
    print("\n--- Especialización por valor de suma ---")
    for sum_val in sorted(sum_expert_map.keys()):
        experts = sum_expert_map[sum_val]
        most_common = max(set(experts), key=experts.count)
        print(f"Suma = {sum_val:2d}: Mayoría usa Experto {most_common} "
              f"({experts.count(most_common)}/{len(experts)} casos)")
    
    return sum_expert_map

if __name__ == "__main__":
    run_experiment_mod10()