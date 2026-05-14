import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_experiment.eval_routing import load_model, analyze_routing
from modular_experiment.eval_multiplication import analyze_routing_mul, compute_entropy

def load_sum_model(model_path="models_shared/model_modulo_5_experts_4.pt"):
    """Cargar modelo de suma modular"""
    if Path(model_path).exists():
        model, modulo = load_model(model_path, num_experts=4)
        return model, modulo
    return None, None

def load_mul_model(model_path="models_shared/model_mul_mod5_experts_4.pt"):
    """Cargar modelo de multiplicación modular"""
    if Path(model_path).exists():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from modular_experiment.model import ModularMoE
        model = ModularMoE(input_dim=2, output_dim=1, num_experts=4, hidden_dim=64)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, checkpoint.get('modulo', None)
    return None, None

def plot_comparative_matrices(sum_routing, mul_routing, modulo=5):
    """Comparar matrices de enrutamiento lado a lado"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Suma modular
    ax1 = axes[0]
    im1 = ax1.imshow(sum_routing, cmap='tab10', vmin=0, vmax=3)
    ax1.set_title(f'Suma Modular (mod {modulo})\nEntropía: {compute_entropy(sum_routing):.3f}', fontsize=12)
    ax1.set_xlabel('b', fontsize=11)
    ax1.set_ylabel('a', fontsize=11)
    for i in range(modulo):
        for j in range(modulo):
            ax1.text(j, i, sum_routing[i, j], ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=ax1, ticks=range(4), label='Expert ID')
    
    # Multiplicación modular
    ax2 = axes[1]
    im2 = ax2.imshow(mul_routing, cmap='tab10', vmin=0, vmax=3)
    entropy_mul = compute_entropy(mul_routing)
    ax2.set_title(f'Multiplicación Modular (mod {modulo})\nEntropía: {entropy_mul:.3f}', fontsize=12)
    ax2.set_xlabel('b', fontsize=11)
    ax2.set_ylabel('a', fontsize=11)
    for i in range(modulo):
        for j in range(modulo):
            ax2.text(j, i, mul_routing[i, j], ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax2, ticks=range(4), label='Expert ID')
    
    plt.suptitle(f'Comparativa: Suma vs Multiplicación Modular (mod {modulo})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/sum_vs_mul_mod{modulo}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return entropy_mul

def analyze_expert_specialization(routing_matrix, operation_name, modulo=5):
    """Analizar especialización de expertos por tipo de operación"""
    
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE ESPECIALIZACIÓN - {operation_name}")
    print(f"{'='*50}")
    
    # Distribución de expertos
    unique, counts = np.unique(routing_matrix, return_counts=True)
    print(f"\nDistribución de expertos:")
    for exp, count in zip(unique, counts):
        print(f"  Experto {exp}: {count/25*100:.1f}% ({count} muestras)")
    
    # Entropía
    entropy = compute_entropy(routing_matrix)
    print(f"\nEntropía: {entropy:.4f} bits")
    
    # Patrones por fila/columna
    print(f"\nPatrón por filas (a constante):")
    for i in range(modulo):
        row_experts = routing_matrix[i, :]
        most_common = Counter(row_experts).most_common(1)[0][0]
        print(f"  Fila {i}: Mayoría Experto {most_common}")
    
    # Simetría
    is_symmetric = np.all(routing_matrix == routing_matrix.T)
    print(f"\nMatriz simétrica: {is_symmetric}")
    
    return entropy

def analyze_special_cases_sum(routing_matrix, modulo=5):
    """Analizar casos especiales en suma modular"""
    
    print(f"\n{'='*50}")
    print("CASOS ESPECIALES - SUMA MODULAR")
    print(f"{'='*50}")
    
    # Diagonal principal (a = b)
    diagonal = [routing_matrix[i, i] for i in range(modulo)]
    print(f"Diagonal principal (a=b): {diagonal}")
    
    # Casos con cero
    zero_cases = [routing_matrix[0, j] for j in range(modulo)]
    print(f"Casos con a=0: {zero_cases}")
    
    # Casos con resultado módulo específico
    print(f"\nRelación con resultado (a+b mod {modulo}):")
    for target in range(modulo):
        positions = [(i, j) for i in range(modulo) for j in range(modulo) 
                    if (i + j) % modulo == target]
        if positions:
            experts = [routing_matrix[i, j] for i, j in positions]
            most_common = Counter(experts).most_common(1)[0][0]
            print(f"  Suma = {target}: Mayoría Experto {most_common}")

def analyze_special_cases_mul(routing_matrix, modulo=5):
    """Analizar casos especiales en multiplicación modular"""
    
    print(f"\n{'='*50}")
    print("CASOS ESPECIALES - MULTIPLICACIÓN MODULAR")
    print(f"{'='*50}")
    
    # Diagonal principal (a = b)
    diagonal = [routing_matrix[i, i] for i in range(modulo)]
    print(f"Diagonal principal (a=b): {diagonal}")
    
    # Casos con cero (absorbente)
    zero_cases = []
    for i in range(modulo):
        for j in range(modulo):
            if (i * j) % modulo == 0:
                zero_cases.append(routing_matrix[i, j])
    zero_experts = Counter(zero_cases)
    print(f"Casos con producto = 0: {dict(zero_experts)}")
    
    # Casos con uno (neutro)
    one_cases = []
    for i in range(modulo):
        for j in range(modulo):
            if (i * j) % modulo == 1:
                one_cases.append(routing_matrix[i, j])
    if one_cases:
        one_experts = Counter(one_cases)
        print(f"Casos con producto = 1: {dict(one_experts)}")
    
    # Casos con resultado específico
    print(f"\nRelación con producto (a×b mod {modulo}):")
    for target in range(modulo):
        positions = [(i, j) for i in range(modulo) for j in range(modulo) 
                    if (i * j) % modulo == target]
        if positions:
            experts = [routing_matrix[i, j] for i, j in positions]
            most_common = Counter(experts).most_common(1)[0][0]
            print(f"  Producto = {target}: Mayoría Experto {most_common}")

def plot_entropy_comparison():
    """Visualizar comparación de entropías entre operaciones"""
    
    # Datos de experimentos anteriores
    operations = ['Suma (mod 5)', 'Multiplicación (mod 5)', 
                  'Suma (mod 6)', 'Suma (mod 7)', 'Suma (mod 10)']
    entropies = [0.440, 0.0, 0.323, 0.447, 0.519]  # 0.0 es placeholder
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(operations, entropies, color=['blue', 'red', 'green', 'orange', 'purple'])
    ax.set_ylabel('Entropía (bits)', fontsize=12)
    ax.set_title('Comparación de Entropía por Operación y Módulo', fontsize=14)
    ax.set_ylim(0, 1.0)
    
    # Añadir valores en las barras
    for bar, val in zip(bars, entropies):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('results/entropy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Análisis comparativo completo"""
    
    os.makedirs("results", exist_ok=True)
    
    print("="*60)
    print("🔬 ANÁLISIS COMPARATIVO: SUMA vs MULTIPLICACIÓN MODULAR")
    print("="*60)
    
    # Cargar modelo de suma (mod 5)
    print("\n📊 Cargando modelo de SUMA modular (mod 5)...")
    sum_model, sum_mod = load_sum_model()
    
    if sum_model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sum_model = sum_model.to(device)
        sum_routing, _ = analyze_routing(sum_model, max_num=4, modulo=5, device=device)
        
        # Análisis de suma
        sum_entropy = analyze_expert_specialization(sum_routing, "SUMA MODULAR (mod 5)")
        analyze_special_cases_sum(sum_routing, modulo=5)
    else:
        print("  ⚠️ Modelo de suma no encontrado. Ejecuta primero:")
        print("     python -m modular_experiment.train --modulo 5")
        sum_routing = None
    
    # Cargar modelo de multiplicación (mod 5)
    print("\n📊 Cargando modelo de MULTIPLICACIÓN modular (mod 5)...")
    mul_model, mul_mod = load_mul_model()
    
    if mul_model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mul_routing, _ = analyze_routing_mul(mul_model, max_num=4, modulo=5, device=device)
        
        # Análisis de multiplicación
        mul_entropy = analyze_expert_specialization(mul_routing, "MULTIPLICACIÓN MODULAR (mod 5)")
        analyze_special_cases_mul(mul_routing, modulo=5)
    else:
        print("  ⚠️ Modelo de multiplicación no encontrado. Ejecuta primero:")
        print("     python -m modular_experiment.train_multiplication --modulo 5")
        mul_routing = None
    
    # Comparación visual
    if sum_routing is not None and mul_routing is not None:
        print("\n📈 Generando visualizaciones comparativas...")
        plot_comparative_matrices(sum_routing, mul_routing, modulo=5)
        
        # Comparación numérica
        print("\n" + "="*50)
        print("COMPARACIÓN NUMÉRICA")
        print("="*50)
        print(f"Suma - Entropía: {sum_entropy:.4f}")
        print(f"Multiplicación - Entropía: {mul_entropy:.4f}")
        print(f"Diferencia: {abs(sum_entropy - mul_entropy):.4f}")
        
        if mul_entropy > sum_entropy:
            print("✅ La multiplicación tiene MAYOR entropía (menos especialización)")
        else:
            print("✅ La multiplicación tiene MENOR entropía (más especialización)")
    
    # Gráfico comparativo general
    plot_entropy_comparison()
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("📁 Resultados guardados en carpeta 'results/'")
    print("   - sum_vs_mul_mod5.png")
    print("   - entropy_comparison.png")
    print("="*60)

if __name__ == "__main__":
    main()