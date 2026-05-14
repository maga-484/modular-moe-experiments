# Crea este archivo para ejecutar después
# modular_experiment/analyze_mod15.py
import numpy as np
import matplotlib.pyplot as plt

def quick_analysis(routing_matrix, modulo=15):
    print(f"\n📊 Análisis rápido módulo {modulo}")
    
    # 1. Estadísticas básicas
    unique, counts = np.unique(routing_matrix, return_counts=True)
    print(f"Expertos activos: {len(unique)}/{np.max(routing_matrix)+1}")
    
    # 2. ¿Hay patrón diagonal?
    diagonal = [routing_matrix[i,i] for i in range(min(5, modulo))]
    print(f"Diagonal (primeros 5): {diagonal}")
    
    # 3. Simetría
    is_symmetric = np.all(routing_matrix == routing_matrix.T)
    print(f"Matriz simétrica: {is_symmetric}")
    
    # 4. Entropía
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    print(f"Entropía: {entropy:.3f} bits")