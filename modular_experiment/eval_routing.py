import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.dataset import AllCombinationsDataset
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

def analyze_routing(model, max_num=4, modulo=None, device='cpu'):
    """Analizar el patrón de enrutamiento para todas las combinaciones"""
    dataset = AllCombinationsDataset(max_num=max_num, modulo=modulo)
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

def compute_expert_specialization(model, max_num=4, modulo=None, device='cpu'):
    """Calcular métricas de especialización de expertos"""
    dataset = AllCombinationsDataset(max_num=max_num, modulo=modulo)
    
    # Contar cuántas muestras asigna cada experto
    expert_counts = np.zeros(model.num_experts)
    # Calcular entropía del enrutamiento
    total_entropy = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, (a, b) = dataset[i]
            x = x.unsqueeze(0).to(device)
            _, gate_weights = model(x)
            expert_idx = torch.argmax(gate_weights, dim=1).item()
            expert_counts[expert_idx] += 1
            
            # Entropía de los pesos del gate
            probs = gate_weights.squeeze().cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            total_entropy += entropy
    
    # Distribución de expertos
    expert_distribution = expert_counts / len(dataset)
    avg_entropy = total_entropy / len(dataset)
    
    return {
        'expert_distribution': expert_distribution,
        'avg_entropy': avg_entropy,
        'expert_counts': expert_counts
    }

def plot_routing_matrix(routing_matrix, modulo=None, save_path=None):
    """Visualizar matriz de enrutamiento"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear heatmap
    im = ax.imshow(routing_matrix, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
    
    # Configurar ejes
    ax.set_xticks(np.arange(len(routing_matrix)))
    ax.set_yticks(np.arange(len(routing_matrix)))
    ax.set_xticklabels(np.arange(len(routing_matrix)))
    ax.set_yticklabels(np.arange(len(routing_matrix)))
    
    # Añadir números en cada celda
    for i in range(len(routing_matrix)):
        for j in range(len(routing_matrix)):
            text = ax.text(j, i, routing_matrix[i, j],
                          ha="center", va="center", color="white",
                          fontsize=12, fontweight='bold')
    
    ax.set_xlabel('b', fontsize=14)
    ax.set_ylabel('a', fontsize=14)
    title = f'Routing Decisions - Modular Addition'
    if modulo is not None:
        title += f' (mod {modulo})'
    else:
        title += ' (normal)'
    ax.set_title(title, fontsize=16)
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ticks=range(int(np.max(routing_matrix)) + 1))
    cbar.set_label('Expert ID', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_gate_weights_heatmap(gate_weight_matrix, expert_id, modulo=None, save_path=None):
    """Visualizar pesos del gate para un experto específico"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extraer pesos para el experto
    weights = gate_weight_matrix[:, :, expert_id]
    
    im = ax.imshow(weights, cmap='viridis', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(weights)))
    ax.set_yticks(np.arange(len(weights)))
    ax.set_xticklabels(np.arange(len(weights)))
    ax.set_yticklabels(np.arange(len(weights)))
    
    # Añadir valores
    for i in range(len(weights)):
        for j in range(len(weights)):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                          ha="center", va="center", color="white" if weights[i, j] < 0.5 else "black",
                          fontsize=10)
    
    ax.set_xlabel('b', fontsize=14)
    ax.set_ylabel('a', fontsize=14)
    title = f'Gate Weights for Expert {expert_id}'
    if modulo is not None:
        title += f' (mod {modulo})'
    ax.set_title(title, fontsize=16)
    
    plt.colorbar(im, label='Gate Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Configuración
    model_path_normal = "models_shared/model_modulo_none_experts_4.pt"
    model_path_mod5 = "models_shared/model_modulo_5_experts_4.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear directorio para resultados
    os.makedirs("results/modular_experiment", exist_ok=True)
    
    print("=" * 60)
    print("ANÁLISIS DE ENRUTAMIENTO - SUMA NORMAL (sin módulo)")
    print("=" * 60)
    
    # Analizar modelo sin módulo
    if Path(model_path_normal).exists():
        model, modulo = load_model(model_path_normal, num_experts=4)
        model = model.to(device)
        
        routing_matrix, gate_weights = analyze_routing(model, max_num=4, modulo=None, device=device)
        print("\nMatriz de enrutamiento (cada celda muestra qué experto se activó):")
        print(routing_matrix)
        
        specialization = compute_expert_specialization(model, max_num=4, modulo=None, device=device)
        print(f"\nDistribución de expertos: {specialization['expert_distribution']}")
        print(f"Entropía promedio del gate: {specialization['avg_entropy']:.4f}")
        
        plot_routing_matrix(routing_matrix, modulo=None, 
                           save_path="results/modular_experiment/routing_normal.png")
    else:
        print(f"Modelo no encontrado: {model_path_normal}")
        print("Primero ejecuta: python modular_experiment/train.py --modulo None")
    
    print("\n" + "=" * 60)
    print("ANÁLISIS DE ENRUTAMIENTO - SUMA MÓDULO 5")
    print("=" * 60)
    
    # Analizar modelo con módulo 5
    if Path(model_path_mod5).exists():
        model, modulo = load_model(model_path_mod5, num_experts=4)
        model = model.to(device)
        
        routing_matrix, gate_weights = analyze_routing(model, max_num=4, modulo=5, device=device)
        print("\nMatriz de enrutamiento (cada celda muestra qué experto se activó):")
        print(routing_matrix)
        
        specialization = compute_expert_specialization(model, max_num=4, modulo=5, device=device)
        print(f"\nDistribución de expertos: {specialization['expert_distribution']}")
        print(f"Entropía promedio del gate: {specialization['avg_entropy']:.4f}")
        
        plot_routing_matrix(routing_matrix, modulo=5,
                           save_path="results/modular_experiment/routing_mod5.png")
        
        # Visualizar pesos del gate para cada experto
        for expert_id in range(model.num_experts):
            plot_gate_weights_heatmap(gate_weights, expert_id, modulo=5,
                                     save_path=f"results/modular_experiment/gate_weights_expert{expert_id}_mod5.png")
    else:
        print(f"Modelo no encontrado: {model_path_mod5}")
        print("Primero ejecuta: python modular_experiment/train.py --modulo 5")

if __name__ == "__main__":
    main()