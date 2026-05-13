import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.train import train
from modular_experiment.eval_routing import analyze_routing, compute_expert_specialization

def run_prime_experiment(modulo, label):
    """Ejecutar experimento para un módulo específico"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: Módulo {modulo} ({label})")
    print(f"{'='*60}")
    
    # Entrenar modelo
    model = train(
        max_num=modulo - 1,
        modulo=modulo,
        num_experts=4,
        epochs=250,
        hidden_dim=64
    )
    
    # Evaluar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    routing_matrix, gate_weights = analyze_routing(
        model, max_num=modulo-1, modulo=modulo, device=device
    )
    
    specialization = compute_expert_specialization(
        model, max_num=modulo-1, modulo=modulo, device=device
    )
    
    print(f"\nMatriz de enrutamiento ({modulo}x{modulo}):")
    print(routing_matrix)
    print(f"\nDistribución de expertos: {specialization['expert_distribution']}")
    print(f"Entropía: {specialization['avg_entropy']:.4f}")
    
    return {
        'modulo': modulo,
        'label': label,
        'routing_matrix': routing_matrix,
        'entropy': specialization['avg_entropy'],
        'distribution': specialization['expert_distribution']
    }

def compare_prime_vs_nonprime():
    """Comparar módulo primo (7) vs no primo (6)"""
    
    os.makedirs("results/experiment_prime_comparison", exist_ok=True)
    
    # Ejecutar experimentos
    results = []
    
    # Módulo 6 (no primo: 6 = 2 × 3)
    results.append(run_prime_experiment(6, "No primo (6)"))
    
    # Módulo 7 (primo)
    results.append(run_prime_experiment(7, "Primo (7)"))
    
    # Comparar visualmente
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, result in enumerate(results):
        routing = result['routing_matrix']
        modulo = result['modulo']
        
        ax = axes[idx]
        im = ax.imshow(routing, cmap='tab10', vmin=0, vmax=3)
        ax.set_title(f"Mod {modulo} ({result['label']})\nEntropy: {result['entropy']:.3f}", 
                    fontsize=14)
        ax.set_xlabel('b', fontsize=12)
        ax.set_ylabel('a', fontsize=12)
        
        # Añadir números
        for i in range(modulo):
            for j in range(modulo):
                if i < len(routing) and j < len(routing):
                    ax.text(j, i, routing[i, j], ha="center", va="center", 
                           color="white", fontsize=10)
        
        plt.colorbar(im, ax=ax, ticks=range(4), label='Expert ID')
    
    plt.suptitle('Prime vs Non-Prime Modular Addition', fontsize=16)
    plt.tight_layout()
    plt.savefig("results/experiment_prime_comparison/prime_vs_nonprime.png", dpi=150)
    plt.show()
    
    # Análisis estadístico
    print("\n" + "="*60)
    print("COMPARACIÓN ESTADÍSTICA")
    print("="*60)
    
    for result in results:
        print(f"\nMódulo {result['modulo']} ({result['label']}):")
        print(f"  - Entropía: {result['entropy']:.4f}")
        print(f"  - Distribución: {result['distribution']}")
        print(f"  - Expertos activos: {np.sum(result['distribution'] > 0.05)} de 4")
        
        # Calcular métrica de "bloquedad"
        routing = result['routing_matrix']
        block_score = calculate_block_score(routing)
        print(f"  - Score de bloques: {block_score:.3f} (mayor = más bloqueado)")

def calculate_block_score(routing_matrix):
    """Calcular qué tan 'bloqueada' es la matriz de enrutamiento"""
    h, w = routing_matrix.shape
    edge_changes = 0
    
    for i in range(h):
        for j in range(w):
            if i > 0 and routing_matrix[i, j] != routing_matrix[i-1, j]:
                edge_changes += 1
            if j > 0 and routing_matrix[i, j] != routing_matrix[i, j-1]:
                edge_changes += 1
    
    max_changes = 2 * h * w
    return 1 - (edge_changes / max_changes)  # 1 = completamente bloqueado, 0 = checkerboard

def analyze_algebraic_structure():
    """Análisis adicional: ver si la especialización respeta subgrupos"""
    
    print("\n" + "="*60)
    print("ANÁLISIS DE ESTRUCTURA ALGEBRAICA")
    print("="*60)
    
    # Para módulo 7 (primo), los subgrupos no triviales no existen
    # Para módulo 6, hay subgrupos: {0,2,4} y {0,3}
    
    # Cargar modelo de módulo 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from modular_experiment.eval_routing import load_model
        
        model_path_6 = "models_shared/model_modulo_6_experts_4.pt"
        model_6, _ = load_model(model_path_6, num_experts=4)
        model_6 = model_6.to(device)
        
        routing_6, _ = analyze_routing(model_6, max_num=5, modulo=6, device=device)
        
        # Verificar si algún experto se especializa en subgrupos
        # Subgrupo de pares: {0,2,4}
        subroup_even = [(0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)]
        
        experts_in_subgroup = [routing_6[a, b] for a, b in subroup_even]
        most_common = max(set(experts_in_subgroup), key=experts_in_subgroup.count)
        
        print(f"Módulo 6 - Subgrupo de pares: mayormente Experto {most_common}")
        print(f"  {experts_in_subgroup.count(most_common)}/{len(subroup_even)} casos")
        
    except FileNotFoundError:
        print("Modelo de módulo 6 no encontrado. Ejecuta primero el experimento.")

if __name__ == "__main__":
    compare_prime_vs_nonprime()
    analyze_algebraic_structure()