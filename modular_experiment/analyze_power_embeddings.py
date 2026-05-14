import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.model import ModularMoE
from modular_experiment.eval_routing import load_model

def extract_power_embeddings(model, max_num=4, modulo=5, device='cpu'):
    """Extraer embeddings para potencia modular"""
    model.eval()
    embeddings = []
    numbers = list(range(max_num + 1))
    
    with torch.no_grad():
        for num in numbers:
            # Para potencia, usamos (num, num) para ver representación
            x = torch.tensor([[num, num]], dtype=torch.float32).to(device)
            gate_logits = model.gate(x)
            embeddings.append(gate_logits.cpu().numpy().squeeze())
    
    return np.array(embeddings), numbers

def visualize_power_embeddings(embeddings, numbers, modulo=5, save_path=None):
    """Visualizar embeddings de potencia"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    ax1 = axes[0]
    scatter1 = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                          c=numbers, cmap='tab10', s=100)
    for i, num in enumerate(numbers):
        ax1.annotate(str(num), embeddings_pca[i], fontsize=12, fontweight='bold')
    ax1.set_title(f'PCA - Potencia Modular (mod {modulo})', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Número')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(numbers)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                          c=numbers, cmap='tab10', s=100)
    for i, num in enumerate(numbers):
        ax2.annotate(str(num), embeddings_tsne[i], fontsize=12, fontweight='bold')
    ax2.set_title(f't-SNE - Potencia Modular (mod {modulo})', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Número')
    
    # 3D
    from mpl_toolkits.mplot3d import Axes3D
    ax3 = fig.add_subplot(133, projection='3d')
    pca3d = PCA(n_components=3)
    embeddings_3d = pca3d.fit_transform(embeddings)
    
    ax3.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
               c=numbers, cmap='tab10', s=80)
    for i, num in enumerate(numbers):
        ax3.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
                str(num), fontsize=10)
    ax3.set_title(f'3D - Potencia Modular (mod {modulo})', fontsize=12)
    
    plt.suptitle(f'Estructura de Embeddings - Potencia Modular', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_power_cycles(model, modulo=5, device='cpu'):
    """Analizar si el gate aprendió los ciclos de potencia"""
    
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE CICLOS - Potencia Modular (mod {modulo})")
    print(f"{'='*50}")
    
    # Calcular resultados de potencia
    results = {}
    for a in range(modulo):
        for b in range(1, modulo):
            result = pow(a, b, modulo)
            results[(a, b)] = result
    
    # Verificar estructura de ciclos
    print("\nCiclos de potencia:")
    for a in range(modulo):
        cycle = []
        val = a
        for _ in range(modulo):
            cycle.append(val)
            val = pow(val, a, modulo) if a > 0 else 0
            if val in cycle:
                break
        print(f"  Base {a}: ciclo {cycle[:6]}...")
    
    # Verificar qué experto maneja cada resultado
    with torch.no_grad():
        expert_by_result = {}
        for a in range(modulo):
            for b in range(modulo):
                if a == 0 and b == 0:
                    continue  # 0^0 es indefinido
                x = torch.tensor([[a, b]], dtype=torch.float32).to(device)
                _, gate_weights = model(x)
                expert = torch.argmax(gate_weights).item()
                result = pow(a, b, modulo) if modulo else a**b
                if result not in expert_by_result:
                    expert_by_result[result] = []
                expert_by_result[result].append(expert)
    
    print("\nEspecialización por resultado:")
    for result, experts in sorted(expert_by_result.items()):
        from collections import Counter
        counter = Counter(experts)
        most_common = counter.most_common(1)[0]
        print(f"  Resultado {result}: mayormente Experto {most_common[0]} ({most_common[1]/len(experts)*100:.1f}%)")

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Potencia mod 5
    model_path = "models_shared/model_pow_mod5_experts_4.pt"
    if Path(model_path).exists():
        print("🔬 Analizando Potencia Modular (mod 5)...")
        model, modulo = load_model(model_path, num_experts=4)
        model = model.to(device)
        
        embeddings, numbers = extract_power_embeddings(model, max_num=4, modulo=5, device=device)
        visualize_power_embeddings(embeddings, numbers, modulo=5,
                                  save_path='results/power_embeddings_mod5.png')
        analyze_power_cycles(model, modulo=5, device=device)
    else:
        print(f"⚠️ Modelo no encontrado: {model_path}")
    
    # Potencia mod 10 (si existe)
    model_path = "models_shared/model_pow_mod10_experts_4.pt"
    if Path(model_path).exists():
        print("\n🔬 Analizando Potencia Modular (mod 10)...")
        model, modulo = load_model(model_path, num_experts=4)
        model = model.to(device)
        
        embeddings, numbers = extract_power_embeddings(model, max_num=9, modulo=10, device=device)
        visualize_power_embeddings(embeddings, numbers, modulo=10,
                                  save_path='results/power_embeddings_mod10.png')

if __name__ == "__main__":
    main()