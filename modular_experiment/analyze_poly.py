import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.model import ModularMoE
from modular_experiment.eval_routing import load_model

def extract_poly_embeddings(model, max_num=4, modulo=5, device='cpu'):
    """Extraer embeddings para polinomio (a²+b)"""
    model.eval()
    embeddings = []
    numbers = list(range(max_num + 1))
    
    with torch.no_grad():
        for num in numbers:
            # Usamos (num, num) para representación del número
            x = torch.tensor([[num, num]], dtype=torch.float32).to(device)
            gate_logits = model.gate(x)
            embeddings.append(gate_logits.cpu().numpy().squeeze())
    
    return np.array(embeddings), numbers

def visualize_poly_embeddings(embeddings, numbers, modulo=5, save_path=None):
    """Visualizar embeddings de polinomio"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    ax1 = axes[0]
    scatter1 = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                          c=numbers, cmap='tab10', s=100)
    for i, num in enumerate(numbers):
        ax1.annotate(str(num), embeddings_pca[i], fontsize=12, fontweight='bold')
    ax1.set_title(f'PCA - Polinomio (a²+b) mod {modulo}', fontsize=12)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1, label='Número')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(numbers)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                          c=numbers, cmap='tab10', s=100)
    for i, num in enumerate(numbers):
        ax2.annotate(str(num), embeddings_tsne[i], fontsize=12, fontweight='bold')
    ax2.set_title(f't-SNE - Polinomio (a²+b) mod {modulo}', fontsize=12)
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
    ax3.set_title(f'3D - Polinomio mod {modulo}', fontsize=12)
    
    plt.suptitle(f'Estructura de Embeddings - Polinomio (a²+b)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_poly_specialization(model, modulo=5, device='cpu'):
    """Analizar especialización del polinomio"""
    
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE ESPECIALIZACIÓN - Polinomio (a²+b) mod {modulo}")
    print(f"{'='*50}")
    
    # Calcular resultados reales
    results = {}
    for a in range(modulo):
        for b in range(modulo):
            val = (a*a + b) % modulo
            results[(a, b)] = val
    
    # Verificar qué experto maneja cada resultado
    with torch.no_grad():
        expert_by_result = {}
        for a in range(modulo):
            for b in range(modulo):
                x = torch.tensor([[a, b]], dtype=torch.float32).to(device)
                _, gate_weights = model(x)
                expert = torch.argmax(gate_weights).item()
                val = (a*a + b) % modulo
                if val not in expert_by_result:
                    expert_by_result[val] = []
                expert_by_result[val].append(expert)
    
    print("\nEspecialización por resultado:")
    for result in sorted(expert_by_result.keys()):
        counter = Counter(expert_by_result[result])
        most_common = counter.most_common(1)[0]
        print(f"  Resultado {result}: mayormente Experto {most_common[0]} ({most_common[1]/len(expert_by_result[result])*100:.1f}%)")
    
    # Analizar casos especiales
    print("\nCasos especiales:")
    
    # Casos con a=0
    zero_a_results = []
    for b in range(modulo):
        val = (0 + b) % modulo
        zero_a_results.append(val)
    print(f"  a=0: resultados {zero_a_results}")
    
    # Casos con a=1
    one_a_results = []
    for b in range(modulo):
        val = (1 + b) % modulo
        one_a_results.append(val)
    print(f"  a=1: resultados {one_a_results}")
    
    # Casos donde a² es significativo
    print(f"\nValores de a² mod {modulo}:")
    for a in range(modulo):
        print(f"  a={a}: a²≡{a*a % modulo}")

def plot_poly_difficulty_comparison():
    """Comparar dificultad entre operaciones"""
    
    operations = ['Potencia', 'Multiplicación', 'Polinomio', 'Suma']
    mod5_loss = [0.000040, 0.000195, 0.000832, 0.00134]
    mod10_loss = [0.000918, 0.000801, 0.0420, 0.000954]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Módulo 5
    ax1 = axes[0]
    bars1 = ax1.bar(operations, mod5_loss, color=['green', 'blue', 'orange', 'red'])
    ax1.set_ylabel('Mejor Loss (log scale)')
    ax1.set_title('Módulo 5')
    ax1.set_yscale('log')
    for bar, val in zip(bars1, mod5_loss):
        ax1.text(bar.get_x() + bar.get_width()/2, val*1.5, f'{val:.6f}', ha='center', fontsize=9)
    
    # Módulo 10
    ax2 = axes[1]
    bars2 = ax2.bar(operations, mod10_loss, color=['green', 'blue', 'orange', 'red'])
    ax2.set_ylabel('Mejor Loss (log scale)')
    ax2.set_title('Módulo 10')
    ax2.set_yscale('log')
    for bar, val in zip(bars2, mod10_loss):
        ax2.text(bar.get_x() + bar.get_width()/2, val*1.5, f'{val:.6f}', ha='center', fontsize=9)
    
    plt.suptitle('Comparativa de Dificultad por Operación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/poly_difficulty_comparison.png', dpi=150)
    plt.show()

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Polinomio mod 5
    model_path = "models_shared/model_poly_mod5_experts_4.pt"
    if Path(model_path).exists():
        print("🔬 Analizando Polinomio (a²+b) mod 5...")
        model, modulo = load_model(model_path, num_experts=4)
        model = model.to(device)
        
        embeddings, numbers = extract_poly_embeddings(model, max_num=4, modulo=5, device=device)
        visualize_poly_embeddings(embeddings, numbers, modulo=5,
                                  save_path='results/poly_embeddings_mod5.png')
        analyze_poly_specialization(model, modulo=5, device=device)
    else:
        print(f"⚠️ Modelo no encontrado: {model_path}")
    
    # Polinomio mod 10 (si existe)
    model_path = "models_shared/model_poly_mod10_experts_4.pt"
    if Path(model_path).exists():
        print("\n🔬 Analizando Polinomio (a²+b) mod 10...")
        model, modulo = load_model(model_path, num_experts=4)
        model = model.to(device)
        
        embeddings, numbers = extract_poly_embeddings(model, max_num=9, modulo=10, device=device)
        visualize_poly_embeddings(embeddings, numbers, modulo=10,
                                  save_path='results/poly_embeddings_mod10.png')
        analyze_poly_specialization(model, modulo=10, device=device)
    
    # Gráfico comparativo
    plot_poly_difficulty_comparison()

if __name__ == "__main__":
    main()