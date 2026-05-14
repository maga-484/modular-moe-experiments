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

def extract_gate_embeddings(model, max_num=4, device='cpu'):
    """
    Extraer los pesos del gate para cada número (a,a)
    para entender cómo el gate representa cada valor
    """
    model.eval()
    embeddings = []
    numbers = list(range(max_num + 1))
    
    with torch.no_grad():
        for num in numbers:
            # Usamos pares (num, num) para ver representación del número
            x = torch.tensor([[num, num]], dtype=torch.float32).to(device)
            
            # Obtenemos los pesos del gate antes del softmax (logits)
            # Para ver la representación interna
            gate_logits = model.gate(x)
            embeddings.append(gate_logits.cpu().numpy().squeeze())
    
    return np.array(embeddings), numbers

def extract_pair_embeddings(model, max_num=4, device='cpu'):
    """
    Extraer embeddings para todos los pares (a,b)
    para ver cómo el gate representa combinaciones
    """
    model.eval()
    embeddings = []
    pairs = []
    
    with torch.no_grad():
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                x = torch.tensor([[a, b]], dtype=torch.float32).to(device)
                gate_logits = model.gate(x)
                embeddings.append(gate_logits.cpu().numpy().squeeze())
                pairs.append((a, b))
    
    return np.array(embeddings), pairs

def visualize_number_embeddings(embeddings, numbers, operation_name, modulo, save_path=None):
    """Visualizar cómo se representan los números individuales"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    ax1 = axes[0]
    scatter1 = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                          c=numbers, cmap='tab10', s=100, alpha=0.7)
    for i, num in enumerate(numbers):
        ax1.annotate(str(num), embeddings_pca[i], fontsize=12, fontweight='bold',
                    ha='center', va='center')
    ax1.set_title(f'PCA - {operation_name} (mod {modulo})', fontsize=12)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1, label='Número')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(numbers)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                          c=numbers, cmap='tab10', s=100, alpha=0.7)
    for i, num in enumerate(numbers):
        ax2.annotate(str(num), embeddings_tsne[i], fontsize=12, fontweight='bold',
                    ha='center', va='center')
    ax2.set_title(f't-SNE - {operation_name} (mod {modulo})', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2, label='Número')
    
    plt.suptitle(f'Embedding Structure - {operation_name} Modular', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_pair_embeddings(embeddings, pairs, operation_name, modulo, max_num=4, save_path=None):
    """Visualizar embeddings de pares como mapa de calor de similitud"""
    
    # Calcular matriz de similitud entre pares
    n_pairs = len(embeddings)
    similarity_matrix = np.zeros((max_num+1, max_num+1, max_num+1, max_num+1))
    
    # Normalizar embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calcular similitud coseno entre pares
    for i, (a1, b1) in enumerate(pairs):
        for j, (a2, b2) in enumerate(pairs):
            sim = np.dot(embeddings_norm[i], embeddings_norm[j])
            similarity_matrix[a1, b1, a2, b2] = sim
    
    # Visualizar similitud promedio por número
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similitud por número (promedio sobre b)
    num_similarity = np.zeros((max_num+1, max_num+1))
    for a1 in range(max_num+1):
        for a2 in range(max_num+1):
            sim_sum = 0
            count = 0
            for b1 in range(max_num+1):
                for b2 in range(max_num+1):
                    sim_sum += similarity_matrix[a1, b1, a2, b2]
                    count += 1
            num_similarity[a1, a2] = sim_sum / count
    
    ax1 = axes[0]
    im1 = ax1.imshow(num_similarity, cmap='viridis', vmin=-1, vmax=1)
    ax1.set_title(f'Similarity by First Number - {operation_name} (mod {modulo})', fontsize=10)
    ax1.set_xlabel('a2')
    ax1.set_ylabel('a1')
    plt.colorbar(im1, ax=ax1, label='Cosine Similarity')
    
    # Similitud por resultado
    # Calcular resultado esperado para cada par
    if operation_name == "Suma":
        results = [(a + b) % modulo for a in range(max_num+1) for b in range(max_num+1)]
    else:  # Multiplicación
        results = [(a * b) % modulo for a in range(max_num+1) for b in range(max_num+1)]
    
    result_similarity = {}
    unique_results = list(set(results))
    
    for r1 in unique_results:
        for r2 in unique_results:
            # Promedio de similitud entre pares con resultado r1 y r2
            sim_sum = 0
            count = 0
            for i, (a1, b1) in enumerate(pairs):
                for j, (a2, b2) in enumerate(pairs):
                    if results[i] == r1 and results[j] == r2:
                        sim_sum += np.dot(embeddings_norm[i], embeddings_norm[j])
                        count += 1
            if count > 0:
                result_similarity[(r1, r2)] = sim_sum / count
    
    # Crear matriz de similitud por resultado
    result_matrix = np.zeros((len(unique_results), len(unique_results)))
    for i, r1 in enumerate(unique_results):
        for j, r2 in enumerate(unique_results):
            result_matrix[i, j] = result_similarity.get((r1, r2), 0)
    
    ax2 = axes[1]
    im2 = ax2.imshow(result_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax2.set_title(f'Similarity by Result - {operation_name} (mod {modulo})', fontsize=10)
    ax2.set_xlabel('Resultado 2')
    ax2.set_ylabel('Resultado 1')
    ax2.set_xticks(range(len(unique_results)))
    ax2.set_yticks(range(len(unique_results)))
    ax2.set_xticklabels(unique_results)
    ax2.set_yticklabels(unique_results)
    plt.colorbar(im2, ax=ax2, label='Cosine Similarity')
    
    plt.suptitle(f'Pair Embedding Similarity - {operation_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_gate_weights_pattern(model, operation_name, modulo, max_num=4, device='cpu'):
    """Analizar patrón de pesos del gate para cada número"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    numbers = list(range(max_num + 1))
    
    for idx, num in enumerate(numbers[:6]):  # Máximo 6 números
        ax = axes[idx // 3, idx % 3]
        
        # Obtener pesos del gate para (num, b) variando b
        b_vals = list(range(max_num + 1))
        gate_weights = []
        
        with torch.no_grad():
            for b in b_vals:
                x = torch.tensor([[num, b]], dtype=torch.float32).to(device)
                _, weights = model(x)
                gate_weights.append(weights.cpu().numpy().squeeze())
        
        gate_weights = np.array(gate_weights)
        
        # Plot
        for expert in range(gate_weights.shape[1]):
            ax.plot(b_vals, gate_weights[:, expert], 'o-', label=f'Expert {expert}', alpha=0.7)
        
        ax.set_title(f'a = {num}', fontsize=12)
        ax.set_xlabel('b')
        ax.set_ylabel('Gate Weight')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gate Weight Patterns - {operation_name} (mod {modulo})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/gate_patterns_{operation_name.lower()}_mod{modulo}.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Análisis completo de embeddings para suma y multiplicación"""
    
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuraciones a analizar
    experiments = [
        ("Suma", "models_shared/model_modulo_5_experts_4.pt", 5),
        ("Multiplicación", "models_shared/model_mul_mod5_experts_4.pt", 5),
        ("Suma", "models_shared/model_modulo_10_experts_4.pt", 10),
        ("Multiplicación", "models_shared/model_mul_mod10_experts_4.pt", 10),
    ]
    
    for op_name, model_path, modulo in experiments:
        print(f"\n{'='*50}")
        print(f"Analizando {op_name} Modular (mod {modulo})")
        print(f"{'='*50}")
        
        if not Path(model_path).exists():
            print(f"  ⚠️ Modelo no encontrado: {model_path}")
            continue
        
        # Cargar modelo
        model, _ = load_model(model_path, num_experts=4)
        model = model.to(device)
        
        # 1. Embeddings de números individuales
        print("  Extrayendo embeddings de números...")
        embeddings, numbers = extract_gate_embeddings(model, max_num=modulo-1, device=device)
        visualize_number_embeddings(embeddings, numbers, op_name, modulo,
                                   save_path=f'results/embeddings_numbers_{op_name.lower()}_mod{modulo}.png')
        
        # 2. Embeddings de pares
        print("  Extrayendo embeddings de pares...")
        pair_embeddings, pairs = extract_pair_embeddings(model, max_num=modulo-1, device=device)
        visualize_pair_embeddings(pair_embeddings, pairs, op_name, modulo, max_num=modulo-1,
                                 save_path=f'results/embeddings_pairs_{op_name.lower()}_mod{modulo}.png')
        
        # 3. Patrones de pesos del gate
        print("  Analizando patrones de gate...")
        analyze_gate_weights_pattern(model, op_name, modulo, max_num=modulo-1, device=device)
    
    print("\n" + "="*50)
    print("✅ Análisis de embeddings completado!")
    print("📁 Resultados guardados en carpeta 'results/'")
    print("   - embeddings_numbers_*.png")
    print("   - embeddings_pairs_*.png")
    print("   - gate_patterns_*.png")
    print("="*50)

if __name__ == "__main__":
    main()