import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset import ModularAdditionDataset
from modular_experiment.model import ModularMoE
from modular_experiment.train import train
from modular_experiment.eval_routing import analyze_routing, compute_expert_specialization

def run_experiment_8experts():
    """Experimento con 8 expertos (módulo 5)"""
    
    os.makedirs("results/experiment_8experts", exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENTO 2: 8 EXPERTOS (MÓDULO 5)")
    print("=" * 60)
    
    # Entrenar con 8 expertos
    model = train(
        max_num=4,
        modulo=5,
        num_experts=8,  # ← 8 expertos
        epochs=300,
        hidden_dim=64
    )
    
    # Evaluar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    routing_matrix, gate_weights = analyze_routing(
        model, max_num=4, modulo=5, device=device
    )
    
    specialization = compute_expert_specialization(
        model, max_num=4, modulo=5, device=device
    )
    
    print(f"\nMatriz de enrutamiento (5x5):")
    print(routing_matrix)
    print(f"\nDistribución de expertos: {specialization['expert_distribution']}")
    print(f"Entropía: {specialization['avg_entropy']:.4f}")
    print(f"Expertos activos: {np.sum(specialization['expert_distribution'] > 0.01)} de 8")
    
    # Visualizar comparación con 4 expertos
    visualize_comparison(routing_matrix, specialization)
    
    return model, routing_matrix, specialization

def visualize_comparison(routing_matrix_8, specialization_8):
    """Comparar con resultados anteriores de 4 expertos"""
    
    # Datos del experimento anterior (4 expertos)
    routing_4 = np.array([
        [0, 0, 0, 3, 3],
        [0, 0, 3, 3, 2],
        [0, 3, 3, 2, 2],
        [3, 3, 2, 2, 2],
        [3, 2, 2, 2, 2]
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz con 4 expertos
    im1 = axes[0].imshow(routing_4, cmap='tab10', vmin=0, vmax=7)
    axes[0].set_title('4 Experts\nEntropy: 0.440', fontsize=14)
    axes[0].set_xlabel('b')
    axes[0].set_ylabel('a')
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, routing_4[i, j], ha="center", va="center", color="white")
    
    # Matriz con 8 expertos
    im2 = axes[1].imshow(routing_matrix_8, cmap='tab10', vmin=0, vmax=7)
    axes[1].set_title(f'8 Experts\nEntropy: {specialization_8["avg_entropy"]:.3f}', fontsize=14)
    axes[1].set_xlabel('b')
    axes[1].set_ylabel('a')
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, routing_matrix_8[i, j], ha="center", va="center", color="white")
    
    plt.tight_layout()
    plt.savefig("results/experiment_8experts/comparison_4vs8.png", dpi=150)
    plt.show()
    
    # Mostrar especialización de cada experto
    print("\n--- Especialización de cada experto ---")
    for exp_id in range(8):
        if specialization_8['expert_distribution'][exp_id] > 0.01:
            print(f"Experto {exp_id}: {specialization_8['expert_distribution'][exp_id]*100:.1f}% de las muestras")

if __name__ == "__main__":
    run_experiment_8experts()