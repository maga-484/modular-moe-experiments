import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_rings_analysis(routing_matrix, modulo=15):
    """Visualización especializada para los anillos concéntricos"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Matriz de enrutamiento original
    ax1 = axes[0, 0]
    im1 = ax1.imshow(routing_matrix, cmap='tab10', vmin=0, vmax=7)
    ax1.set_title('Routing Matrix - Mod 15\n(Anillos Concéntricos)', fontsize=14)
    ax1.set_xlabel('b (second number)', fontsize=12)
    ax1.set_ylabel('a (first number)', fontsize=12)
    
    # Marcar las regiones
    ax1.axhline(y=4.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=8.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=4.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=8.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Añadir colorbar
    plt.colorbar(im1, ax=ax1, ticks=range(8), label='Expert ID')
    
    # 2. Heatmap de densidad de expertos
    ax2 = axes[0, 1]
    unique_experts = np.unique(routing_matrix)
    expert_density = np.zeros((modulo, modulo, len(unique_experts)))
    
    for i, exp in enumerate(unique_experts):
        expert_density[:, :, i] = (routing_matrix == exp).astype(float)
    
    # Mostrar experto dominante por región
    dominant_expert = np.argmax(expert_density, axis=2)
    im2 = ax2.imshow(dominant_expert, cmap='tab10', vmin=0, vmax=7)
    ax2.set_title('Dominant Expert by Region', fontsize=14)
    ax2.set_xlabel('b', fontsize=12)
    ax2.set_ylabel('a', fontsize=12)
    plt.colorbar(im2, ax=ax2, ticks=range(8), label='Expert ID')
    
    # 3. Distribución de expertos por diagonal (a+b constante)
    ax3 = axes[0, 2]
    sum_diagonals = {}
    for s in range(2*modulo - 1):
        sum_diagonals[s] = []
    
    for i in range(modulo):
        for j in range(modulo):
            s = i + j
            sum_diagonals[s].append(routing_matrix[i, j])
    
    diagonal_experts = []
    diagonal_sums = []
    for s in range(2*modulo - 1):
        experts_in_diag = sum_diagonals[s]
        # Usar Counter para encontrar el más común
        counter = Counter(experts_in_diag)
        most_common = counter.most_common(1)[0][0]
        diagonal_experts.append(most_common)
        diagonal_sums.append(s)
    
    ax3.plot(diagonal_sums, diagonal_experts, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Sum (a+b)', fontsize=12)
    ax3.set_ylabel('Dominant Expert', fontsize=12)
    ax3.set_title('Expert by Diagonal Sum\n(Anillos)', fontsize=14)
    ax3.set_yticks(range(8))
    ax3.grid(True, alpha=0.3)
    
    # 4. Mapa de calor con etiquetas (primeras 10x10)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(routing_matrix[:10, :10], cmap='tab10', vmin=0, vmax=7)
    # Añadir texto
    for i in range(min(10, modulo)):
        for j in range(min(10, modulo)):
            ax4.text(j, i, routing_matrix[i, j], 
                    ha='center', va='center', 
                    color='white' if routing_matrix[i, j] < 4 else 'black',
                    fontsize=9, fontweight='bold')
    ax4.set_title('Detailed View (First 10x10)', fontsize=14)
    ax4.set_xlabel('b', fontsize=12)
    ax4.set_ylabel('a', fontsize=12)
    plt.colorbar(im4, ax=ax4, ticks=range(8))
    
    # 5. Transición de expertos (gradiente horizontal)
    ax5 = axes[1, 1]
    gradient_h = np.zeros((modulo, modulo-1))
    for i in range(modulo):
        for j in range(modulo-1):
            if routing_matrix[i, j] != routing_matrix[i, j+1]:
                gradient_h[i, j] = 1
    
    ax5.imshow(gradient_h, cmap='Reds', interpolation='nearest', aspect='auto')
    ax5.set_title('Expert Transitions (Horizontal)', fontsize=14)
    ax5.set_xlabel('b', fontsize=12)
    ax5.set_ylabel('a', fontsize=12)
    
    # 6. Análisis de "anillos"
    ax6 = axes[1, 2]
    # Calcular "radio" desde la esquina superior izquierda
    ring_distances = np.zeros((modulo, modulo))
    for i in range(modulo):
        for j in range(modulo):
            ring_distances[i, j] = min(i, j, modulo-1-i, modulo-1-j)
    
    # Correlación entre distancia al borde y experto
    ring_expert_map = {}
    for ring in range(int(np.max(ring_distances)) + 1):
        mask = (ring_distances == ring)
        if np.any(mask):
            experts_in_ring = routing_matrix[mask].flatten()
            counter = Counter(experts_in_ring)
            most_common = counter.most_common(1)[0][0]
            ring_expert_map[ring] = most_common
    
    rings = list(ring_expert_map.keys())
    experts = [ring_expert_map[r] for r in rings]
    
    ax6.plot(rings, experts, 'o-', color='purple', linewidth=2, markersize=10)
    ax6.set_xlabel('Ring Distance from Edge', fontsize=12)
    ax6.set_ylabel('Dominant Expert', fontsize=12)
    ax6.set_title('Expert by Ring Layer\n(Análisis de Anillos)', fontsize=14)
    ax6.set_yticks(range(8))
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Módulo 15 - Análisis de Anillos Concéntricos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/mod15_rings_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_ring_layers_comparison(routing_matrix, modulo=15):
    """Comparación visual de las capas de anillos"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Definir las capas de anillos basadas en la matriz real
    layers = [
        (0, "Capa 1: Esquina (Expertos 3,2)", [(0,0), (0,1), (1,0), (1,1), (0,2), (2,0)]),
        (1, "Capa 2: Experto 6", [(2,2), (2,3), (3,2), (3,3), (2,4), (4,2)]),
        (2, "Capa 3: Experto 1", [(4,6), (4,7), (5,6), (5,7), (6,4), (7,4)]),
        (3, "Capa 4: Experto 0", [(6,6), (6,7), (7,6), (7,7), (8,5), (9,5)]),
        (4, "Capa 5: Transición", [(11,0), (11,1), (12,0), (12,1), (0,11), (1,11)]),
    ]
    
    for idx, (layer_id, title, positions) in enumerate(layers[:5]):
        ax = axes[idx // 3, idx % 3]
        
        # Mostrar matriz completa
        im = ax.imshow(routing_matrix, cmap='tab10', vmin=0, vmax=7, alpha=0.6)
        
        # Marcar posiciones de esta capa
        for pos in positions:
            if pos[0] < modulo and pos[1] < modulo:
                rect = Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1, 
                               linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(pos[1], pos[0], str(routing_matrix[pos[0], pos[1]]), 
                       ha='center', va='center', color='red', fontweight='bold', fontsize=8)
        
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('b')
        ax.set_ylabel('a')
        ax.set_xlim(-0.5, modulo-0.5)
        ax.set_ylim(modulo-0.5, -0.5)
    
    plt.suptitle('Visualización de Capas de Anillos - Módulo 15', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/mod15_ring_layers.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_expert_transition_analysis(routing_matrix, modulo=15):
    """Análisis detallado de las transiciones entre expertos"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Mapa de transiciones (dónde cambia el experto)
    ax1 = axes[0]
    transitions = np.zeros((modulo, modulo))
    
    for i in range(modulo):
        for j in range(modulo):
            # Verificar vecinos
            neighbors = 0
            if i > 0 and routing_matrix[i, j] != routing_matrix[i-1, j]:
                neighbors += 1
            if j > 0 and routing_matrix[i, j] != routing_matrix[i, j-1]:
                neighbors += 1
            if i < modulo-1 and routing_matrix[i, j] != routing_matrix[i+1, j]:
                neighbors += 1
            if j < modulo-1 and routing_matrix[i, j] != routing_matrix[i, j+1]:
                neighbors += 1
            transitions[i, j] = neighbors / 4.0  # Normalizar
    
    im1 = ax1.imshow(transitions, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_title('Transition Density Map\n(where experts change)', fontsize=14)
    ax1.set_xlabel('b', fontsize=12)
    ax1.set_ylabel('a', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Transition Probability')
    
    # 2. Perfil de transición a lo largo de la diagonal
    ax2 = axes[1]
    diagonal_transitions = []
    for i in range(modulo-1):
        if routing_matrix[i, i] != routing_matrix[i+1, i+1]:
            diagonal_transitions.append(1)
        else:
            diagonal_transitions.append(0)
    
    ax2.plot(range(len(diagonal_transitions)), diagonal_transitions, 'o-', linewidth=2)
    ax2.set_xlabel('Position along diagonal', fontsize=12)
    ax2.set_ylabel('Transition (0=no, 1=yes)', fontsize=12)
    ax2.set_title('Expert Transitions Along Main Diagonal', fontsize=14)
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Transiciones - Módulo 15', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/mod15_transition_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Función principal"""
    
    # Crear directorio de resultados
    os.makedirs("results", exist_ok=True)
    
    # Matriz de enrutamiento de módulo 15 (de nuestros resultados)
    routing_matrix = np.array([
        [3,3,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [3,3,3,2,2,2,2,2,2,2,2,2,2,2,2],
        [3,3,6,6,6,2,2,2,2,2,2,2,2,2,2],
        [3,3,3,6,6,6,6,2,2,2,2,2,2,2,2],
        [3,3,3,6,6,6,1,1,1,1,2,2,2,2,2],
        [3,3,3,6,6,6,6,1,1,1,1,1,2,2,2],
        [3,3,3,3,0,0,0,6,1,1,1,1,1,1,1],
        [3,3,3,3,0,0,0,0,6,1,1,1,1,1,1],
        [3,3,3,3,3,0,0,0,0,0,1,1,1,1,1],
        [3,3,3,3,3,0,0,0,0,0,0,1,1,1,1],
        [3,3,3,3,3,0,0,0,0,0,0,0,1,1,1],
        [3,3,3,3,3,3,0,0,0,0,0,0,0,0,1],
        [3,3,3,3,3,3,0,0,0,0,0,0,0,0,0],
        [3,3,3,3,3,3,0,0,0,0,0,0,0,0,0],
        [3,3,3,3,3,3,3,0,0,0,0,0,0,0,0]
    ])
    
    modulo = 15
    
    print("\n🔬 Generando visualización de anillos concéntricos...")
    plot_rings_analysis(routing_matrix, modulo=modulo)
    
    print("\n🎨 Generando visualización de capas...")
    plot_ring_layers_comparison(routing_matrix, modulo=modulo)
    
    print("\n📊 Generando análisis de transiciones...")
    plot_expert_transition_analysis(routing_matrix, modulo=modulo)
    
    print("\n✅ Visualizaciones guardadas en 'results/'")
    print("  - results/mod15_rings_analysis.png")
    print("  - results/mod15_ring_layers.png")
    print("  - results/mod15_transition_analysis.png")

if __name__ == "__main__":
    main()