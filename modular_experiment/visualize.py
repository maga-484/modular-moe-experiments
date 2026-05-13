import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.eval_routing import load_model, analyze_routing

def compare_routing_matrices():
    """Comparar lado a lado las matrices de enrutamiento"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Modelo sin módulo
    model_path_normal = "models_shared/model_modulo_none_experts_4.pt"
    if Path(model_path_normal).exists():
        model, _ = load_model(model_path_normal, num_experts=4)
        model = model.to(device)
        routing_normal, _ = analyze_routing(model, max_num=4, modulo=None, device=device)
        
        im1 = axes[0].imshow(routing_normal, cmap='tab10', vmin=0, vmax=9)
        axes[0].set_xticks(np.arange(5))
        axes[0].set_yticks(np.arange(5))
        axes[0].set_xticklabels(np.arange(5))
        axes[0].set_yticklabels(np.arange(5))
        axes[0].set_xlabel('b', fontsize=12)
        axes[0].set_ylabel('a', fontsize=12)
        axes[0].set_title('Suma Normal (sin módulo)', fontsize=14)
        
        # Añadir números
        for i in range(5):
            for j in range(5):
                axes[0].text(j, i, routing_normal[i, j], ha="center", va="center", 
                            color="white", fontsize=12, fontweight='bold')
    
    # Modelo con módulo 5
    model_path_mod5 = "models_shared/model_modulo_5_experts_4.pt"
    if Path(model_path_mod5).exists():
        model, _ = load_model(model_path_mod5, num_experts=4)
        model = model.to(device)
        routing_mod5, _ = analyze_routing(model, max_num=4, modulo=5, device=device)
        
        im2 = axes[1].imshow(routing_mod5, cmap='tab10', vmin=0, vmax=9)
        axes[1].set_xticks(np.arange(5))
        axes[1].set_yticks(np.arange(5))
        axes[1].set_xticklabels(np.arange(5))
        axes[1].set_yticklabels(np.arange(5))
        axes[1].set_xlabel('b', fontsize=12)
        axes[1].set_ylabel('a', fontsize=12)
        axes[1].set_title('Suma Módulo 5', fontsize=14)
        
        # Añadir números
        for i in range(5):
            for j in range(5):
                axes[1].text(j, i, routing_mod5[i, j], ha="center", va="center", 
                            color="white", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("results/modular_experiment/comparison_routing.png", dpi=150, bbox_inches='tight')
    plt.show()

def visualize_predictions():
    """Visualizar predicciones del modelo comparadas con la verdad"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo con módulo 5
    model_path = "models_shared/model_modulo_5_experts_4.pt"
    if not Path(model_path).exists():
        print("Modelo no encontrado. Ejecuta primero el entrenamiento.")
        return
    
    model, _ = load_model(model_path, num_experts=4)
    model = model.to(device)
    
    # Generar todas las combinaciones
    predictions = np.zeros((5, 5))
    ground_truth = np.zeros((5, 5))
    
    with torch.no_grad():
        for a in range(5):
            for b in range(5):
                x = torch.tensor([[a, b]], dtype=torch.float32).to(device)
                output, _ = model(x)
                predictions[a, b] = output.item()
                ground_truth[a, b] = (a + b) % 5
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Predicciones
    im1 = axes[0].imshow(predictions, cmap='viridis')
    axes[0].set_title('Predicciones del Modelo', fontsize=14)
    axes[0].set_xlabel('b')
    axes[0].set_ylabel('a')
    plt.colorbar(im1, ax=axes[0])
    
    # Ground truth
    im2 = axes[1].imshow(ground_truth, cmap='viridis')
    axes[1].set_title('Ground Truth (módulo 5)', fontsize=14)
    axes[1].set_xlabel('b')
    axes[1].set_ylabel('a')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(predictions - ground_truth)
    im3 = axes[2].imshow(error, cmap='Reds')
    axes[2].set_title('Error Absoluto', fontsize=14)
    axes[2].set_xlabel('b')
    axes[2].set_ylabel('a')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("results/modular_experiment/predictions_vs_truth.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    os.makedirs("results/modular_experiment", exist_ok=True)
    
    print("Comparando matrices de enrutamiento...")
    compare_routing_matrices()
    
    print("\nVisualizando predicciones...")
    visualize_predictions()