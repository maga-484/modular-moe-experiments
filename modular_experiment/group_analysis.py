import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean, cityblock, minkowski
from itertools import combinations
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.model import ModularMoE
from modular_experiment.eval_routing import load_model

def extract_number_embeddings(model, max_num=4, device='cpu'):
    """Extraer embedding de cada número usando (num, num)"""
    model.eval()
    embeddings = []
    numbers = list(range(max_num + 1))
    
    with torch.no_grad():
        for num in numbers:
            x = torch.tensor([[num, num]], dtype=torch.float32).to(device)
            # Usar los logits del gate ANTES del softmax
            gate_logits = model.gate(x)
            embeddings.append(gate_logits.cpu().numpy().squeeze())
    
    return np.array(embeddings), numbers

def compute_distances(embeddings, labels, metric='euclidean'):
    """Calcular matriz de distancias entre embeddings"""
    n = len(embeddings)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if metric == 'euclidean':
                distances[i, j] = euclidean(embeddings[i], embeddings[j])
            elif metric == 'manhattan':
                distances[i, j] = cityblock(embeddings[i], embeddings[j])
            elif metric == 'minkowski':
                distances[i, j] = minkowski(embeddings[i], embeddings[j], 3)
    
    return distances

def analyze_group_structure(embeddings, numbers, modulo, operation_name):
    """Analizar si los embeddings respetan la estructura de grupo"""
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE GRUPO - {operation_name} (mod {modulo})")
    print(f"{'='*60}")
    
    # 1. Distancia entre números y sus inversos (para multiplicación)
    if operation_name == "Multiplicación":
        print("\n1. ANÁLISIS DE INVERSOS MULTIPLICATIVOS:")
        for num in numbers[1:]:  # Excepto 0
            # Encontrar inverso tal que (num * inv) % modulo == 1
            inverse = None
            for candidate in numbers[1:]:
                if (num * candidate) % modulo == 1:
                    inverse = candidate
                    break
            
            if inverse is not None:
                dist = euclidean(embeddings[num], embeddings[inverse])
                print(f"  Número {num} ↔ inverso {inverse}: distancia = {dist:.4f}")
    
    # 2. Análisis de subgrupos (para suma)
    if operation_name == "Suma":
        print("\n2. ANÁLISIS DE SUBGRUPOS:")
        # Verificar si pares/impares están separados
        even_nums = [n for n in numbers if n % 2 == 0]
        odd_nums = [n for n in numbers if n % 2 == 1]
        
        # Distancia intra-grupo vs inter-grupo
        intra_even = []
        for i, j in combinations(even_nums, 2):
            intra_even.append(euclidean(embeddings[i], embeddings[j]))
        
        intra_odd = []
        for i, j in combinations(odd_nums, 2):
            intra_odd.append(euclidean(embeddings[i], embeddings[j]))
        
        inter_distances = []
        for e in even_nums:
            for o in odd_nums:
                inter_distances.append(euclidean(embeddings[e], embeddings[o]))
        
        print(f"  Pares: distancia intra-grupo = {np.mean(intra_even):.4f}")
        print(f"  Impares: distancia intra-grupo = {np.mean(intra_odd):.4f}")
        print(f"  Pares-Impares: distancia inter-grupo = {np.mean(inter_distances):.4f}")
        
        if np.mean(inter_distances) > np.mean(intra_even):
            print("  ✅ Subgrupos de pares/impares están separados")
    
    # 3. Conservación de la operación (para multiplicación)
    if operation_name == "Multiplicación" and modulo == 5:
        print("\n3. CONSERVACIÓN DE LA OPERACIÓN:")
        # Verificar si d(a,b) ≈ d(prod(a,b), 1)
        for a in [2, 3, 4]:
            for b in [2, 3, 4]:
                prod = (a * b) % modulo
                if prod in numbers:
                    dist_ab = euclidean(embeddings[a], embeddings[b])
                    dist_prod_1 = euclidean(embeddings[prod], embeddings[1])
                    ratio = dist_ab / (dist_prod_1 + 1e-8)
                    print(f"  d({a},{b})={dist_ab:.4f} vs d({prod},1)={dist_prod_1:.4f} (ratio={ratio:.2f})")

def visualize_3d_embeddings(embeddings, numbers, operation_name, modulo, save_path=None):
    """Visualización 3D de embeddings para ver estructura circular"""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reducir a 3D con PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colores según número
    colors = plt.cm.tab10(np.array(numbers) / max(numbers))
    
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
              c=colors, s=100, alpha=0.7)
    
    # Etiquetas
    for i, num in enumerate(numbers):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
               str(num), fontsize=12, fontweight='bold')
    
    ax.set_title(f'{operation_name} Modular (mod {modulo}) - Embeddings 3D', fontsize=14)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Ajustar ángulo para mejor visualización
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # También mostrar desde diferentes ángulos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    
    angles = [(15, 0), (15, 60), (15, 120)]
    for ax, (elev, azim) in zip(axes, angles):
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                  c=colors, s=80, alpha=0.7)
        for i, num in enumerate(numbers):
            ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                   str(num), fontsize=10)
        ax.set_title(f'Ángulo: elev={elev}°, azim={azim}°')
        ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle(f'{operation_name} mod {modulo} - Múltiples ángulos', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_angles.png'), dpi=150, bbox_inches='tight')
    plt.show()

def compare_distance_metrics(embeddings, numbers, operation_name, modulo):
    """Comparar diferentes métricas de distancia"""
    
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN DE MÉTRICAS - {operation_name} (mod {modulo})")
    print(f"{'='*60}")
    
    metrics = ['euclidean', 'manhattan', 'minkowski']
    results = {}
    
    for metric in metrics:
        distances = compute_distances(embeddings, numbers, metric)
        results[metric] = distances
    
    # Visualizar comparación
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        im = ax.imshow(results[metric], cmap='viridis', aspect='auto')
        ax.set_title(f'{metric.capitalize()} Distance', fontsize=12)
        ax.set_xlabel('Número')
        ax.set_ylabel('Número')
        ax.set_xticks(range(len(numbers)))
        ax.set_yticks(range(len(numbers)))
        ax.set_xticklabels(numbers)
        ax.set_yticklabels(numbers)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Comparativa de Métricas - {operation_name} mod {modulo}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/distance_metrics_{operation_name.lower()}_mod{modulo}.png', dpi=150)
    plt.show()
    
    # Calcular correlación entre métricas
    print("\nCorrelación entre métricas:")
    for m1, m2 in [('euclidean', 'manhattan'), ('euclidean', 'minkowski'), ('manhattan', 'minkowski')]:
        flat1 = results[m1].flatten()
        flat2 = results[m2].flatten()
        corr = np.corrcoef(flat1, flat2)[0, 1]
        print(f"  {m1} vs {m2}: {corr:.4f}")
    
    return results

def predict_result_from_embedding(embeddings, numbers, operation_name, modulo):
    """¿Podemos predecir el resultado a partir de los embeddings?"""
    
    print(f"\n{'='*60}")
    print(f"PREDICCIÓN DESDE EMBEDDINGS - {operation_name} (mod {modulo})")
    print(f"{'='*60}")
    
    # Para multiplicación, ver si embeddings de a y b predicen embedding de producto
    if operation_name == "Multiplicación":
        print("\n¿El embedding de un número predice su inverso?")
        for num in numbers[1:]:
            # Encontrar inverso
            inverse = None
            for candidate in numbers[1:]:
                if (num * candidate) % modulo == 1:
                    inverse = candidate
                    break
            
            if inverse is not None:
                # Usar embedding del inverso para predecir el original
                pred_dist = euclidean(embeddings[num], embeddings[inverse])
                print(f"  Número {num} ↔ inverso {inverse}: distancia = {pred_dist:.4f}")
                
                # Verificar si es el más cercano
                distances = [euclidean(embeddings[num], embeddings[other]) for other in numbers]
                closest = np.argmin(distances[1:]) + 1  # excluir el mismo
                print(f"    Vecino más cercano: {closest} (inverso esperado: {inverse})")
    
    # Análisis de simetría
    print("\nSimetría de la representación:")
    for i, a in enumerate(numbers[:3]):
        for j, b in enumerate(numbers[:3]):
            if a != b:
                dist_ab = euclidean(embeddings[a], embeddings[b])
                dist_ba = euclidean(embeddings[b], embeddings[a])
                print(f"  d({a},{b})={dist_ab:.4f} vs d({b},{a})={dist_ba:.4f}")

def main():
    """Análisis completo de teoría de grupos"""
    
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    experiments = [
        ("Suma", "models_shared/model_modulo_5_experts_4.pt", 5),
        ("Multiplicación", "models_shared/model_mul_mod5_experts_4.pt", 5),
        ("Suma", "models_shared/model_modulo_10_experts_4.pt", 10),
        ("Multiplicación", "models_shared/model_mul_mod10_experts_4.pt", 10),
    ]
    
    for op_name, model_path, modulo in experiments:
        if not Path(model_path).exists():
            print(f"⚠️ Modelo no encontrado: {model_path}")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# {op_name} Modular (mod {modulo})")
        print(f"{'#'*60}")
        
        # Cargar modelo y embeddings
        model, _ = load_model(model_path, num_experts=4)
        model = model.to(device)
        embeddings, numbers = extract_number_embeddings(model, max_num=modulo-1, device=device)
        
        # 1. Análisis de estructura de grupo
        analyze_group_structure(embeddings, numbers, modulo, op_name)
        
        # 2. Visualización 3D
        visualize_3d_embeddings(embeddings, numbers, op_name, modulo,
                               save_path=f'results/3d_embeddings_{op_name.lower()}_mod{modulo}.png')
        
        # 3. Comparación de métricas de distancia
        compare_distance_metrics(embeddings, numbers, op_name, modulo)
        
        # 4. Predicción desde embeddings
        predict_result_from_embedding(embeddings, numbers, op_name, modulo)
    
    print("\n" + "="*60)
    print("✅ Análisis de teoría de grupos completado!")
    print("📁 Resultados en carpeta 'results/'")
    print("   - 3d_embeddings_*.png")
    print("   - distance_metrics_*.png")
    print("="*60)

if __name__ == "__main__":
    main()