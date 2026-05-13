import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.eval_routing import load_model
from modular_experiment.model import ModularMoE

def visualize_expert_functions(model_path="models_shared/model_modulo_5_experts_4.pt"):
    """Visualizar la función que aprendió cada experto"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo
    model, modulo = load_model(model_path, num_experts=4)
    model = model.to(device)
    model.eval()
    
    # Crear grid de entrada
    a_vals = np.linspace(0, 4, 50)
    b_vals = np.linspace(0, 4, 50)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # Calcular salidas de cada experto
    expert_outputs = []
    with torch.no_grad():
        for i in range(len(a_vals)):
            row_outputs = []
            for j in range(len(b_vals)):
                x = torch.tensor([[a_vals[i], b_vals[j]]], dtype=torch.float32).to(device)
                expert_out = model.get_expert_outputs(x)  # (1, num_experts, 1)
                row_outputs.append(expert_out.squeeze().cpu().numpy())
            expert_outputs.append(row_outputs)
    
    expert_outputs = np.array(expert_outputs)  # (50, 50, 4)
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for exp_id in range(4):
        ax = axes[exp_id]
        surf = ax.plot_surface(A, B, expert_outputs[:, :, exp_id], 
                               cmap='viridis', alpha=0.8)
        ax.set_title(f'Expert {exp_id} Function', fontsize=12)
        ax.set_xlabel('a', fontsize=10)
        ax.set_ylabel('b', fontsize=10)
        ax.set_zlabel('Output', fontsize=10)
        fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.suptitle('Expert Functions - Modular Addition (mod 5)', fontsize=14)
    plt.tight_layout()
    plt.savefig("results/experiment_visualize/expert_functions_3d.png", dpi=150)
    plt.show()
    
    # También visualizar como heatmap 2D
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for exp_id in range(4):
        im = axes[exp_id].imshow(expert_outputs[:, :, exp_id].T, 
                                 origin='lower', extent=[0, 4, 0, 4],
                                 cmap='viridis', aspect='auto')
        axes[exp_id].set_title(f'Expert {exp_id} Output', fontsize=12)
        axes[exp_id].set_xlabel('a', fontsize=10)
        axes[exp_id].set_ylabel('b', fontsize=10)
        plt.colorbar(im, ax=axes[exp_id])
    
    plt.suptitle('Expert Functions (Heatmap) - Modular Addition mod 5', fontsize=14)
    plt.tight_layout()
    plt.savefig("results/experiment_visualize/expert_functions_heatmap.png", dpi=150)
    plt.show()
    
    # Analizar patrones
    print("\n--- Análisis de funciones de expertos ---")
    for exp_id in range(4):
        output_range = [expert_outputs[:, :, exp_id].min(), 
                       expert_outputs[:, :, exp_id].max()]
        print(f"Experto {exp_id}: salida en [{output_range[0]:.3f}, {output_range[1]:.3f}]")

def visualize_gate_decision_boundaries(model_path="models_shared/model_modulo_5_experts_4.pt"):
    """Visualizar las fronteras de decisión del gate"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model(model_path, num_experts=4)
    model = model.to(device)
    model.eval()
    
    # Crear grid fino
    a_vals = np.linspace(0, 4, 200)
    b_vals = np.linspace(0, 4, 200)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # Calcular decisiones del gate
    decisions = np.zeros_like(A, dtype=int)
    with torch.no_grad():
        for i in range(len(a_vals)):
            for j in range(len(b_vals)):
                x = torch.tensor([[a_vals[i], b_vals[j]]], dtype=torch.float32).to(device)
                decision = model.get_routing_decision(x)
                decisions[i, j] = decision.item()
    
    # Visualizar
    plt.figure(figsize=(10, 8))
    im = plt.imshow(decisions.T, origin='lower', extent=[0, 4, 0, 4],
                    cmap='tab10', interpolation='nearest')
    plt.colorbar(im, ticks=range(4), label='Expert ID')
    plt.title('Gate Decision Boundaries - Modular Addition (mod 5)', fontsize=14)
    plt.xlabel('a', fontsize=12)
    plt.ylabel('b', fontsize=12)
    
    # Marcar los puntos enteros
    for a in range(5):
        for b in range(5):
            plt.plot(a, b, 'wo', markersize=8, markeredgecolor='black')
            plt.text(a, b, f'{(a+b)%5}', ha='center', va='center', 
                    fontsize=9, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("results/experiment_visualize/gate_boundaries.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    os.makedirs("results/experiment_visualize", exist_ok=True)
    
    print("Visualizando funciones de expertos...")
    visualize_expert_functions()
    
    print("\nVisualizando fronteras de decisión...")
    visualize_gate_decision_boundaries()