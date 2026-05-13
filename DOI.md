## 🔍 ¿Podemos seguir aumentando? Análisis de escalabilidad

## maga https://github.com/maga-484

¡Sí podemos, pero con consideraciones importantes! Analicemos los límites:

### **Resultados de escalado hasta ahora**

| Módulo | Tamaño matriz | Entropía | Tiempo entrenamiento | ¿Viable?        |
| ------ | ------------- | -------- | -------------------- | --------------- |
| 5      | 5×5 (25)      | 0.440    | 4 min                | ✅ Excelente    |
| 6      | 6×6 (36)      | 0.323    | 6 min                | ✅ Excelente    |
| 7      | 7×7 (49)      | 0.447    | 6 min                | ✅ Bueno        |
| 10     | 10×10 (100)   | 0.519    | 6 min                | ✅ Aceptable    |
| 15     | 15×15 (225)   | ?        | ~15 min              | ⚠️ Probable     |
| 20     | 20×20 (400)   | ?        | ~30 min              | ⚠️ Límite CPU   |
| 50     | 50×50 (2500)  | ?        | ~3 horas             | ❌ Necesita GPU |

### **Limitaciones actuales**

1. **Memoria**: Matriz de gate O(n²) crece cuadráticamente
2. **Tiempo**: Escala lineal con n² combinaciones
3. **Especialización**: Entropía aumenta con n (peor especialización)

### **Propuesta para seguir aumentando**

Creemos un experimento de escalabilidad:

```python
# modular_experiment/experiment_scalability.py
import time
import pandas as pd
import matplotlib.pyplot as plt

def test_scalability(modulos=[5,6,7,8,9,10,12,15,20]):
    resultados = []

    for modulo in modulos:
        print(f"\nProbando módulo {modulo}...")

        # Medir tiempo de entrenamiento
        start_time = time.time()

        model = train(
            max_num=modulo-1,
            modulo=modulo,
            num_experts=4,
            epochs=100,  # Menos épocas para prueba rápida
            hidden_dim=64
        )

        train_time = time.time() - start_time

        # Evaluar especialización
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        routing_matrix, _ = analyze_routing(model, max_num=modulo-1, modulo=modulo, device=device)

        # Calcular métricas
        n_experts_activos = len(np.unique(routing_matrix))
        entropia = compute_entropy(routing_matrix)

        resultados.append({
            'modulo': modulo,
            'tamanio': modulo**2,
            'tiempo_min': train_time / 60,
            'expertos_activos': n_experts_activos,
            'entropia': entropia
        })

        print(f"  Tiempo: {train_time/60:.1f} min")
        print(f"  Expertos activos: {n_experts_activos}/4")
        print(f"  Entropía: {entropia:.4f}")

    # Visualizar escalabilidad
    df = pd.DataFrame(resultados)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df['modulo'], df['tiempo_min'], 'o-')
    axes[0].set_xlabel('Módulo (n)')
    axes[0].set_ylabel('Tiempo de entrenamiento (minutos)')
    axes[0].set_title('Escalabilidad Temporal')

    axes[1].plot(df['modulo'], df['entropia'], 'o-', color='red')
    axes[1].set_xlabel('Módulo (n)')
    axes[1].set_ylabel('Entropía de enrutamiento')
    axes[1].set_title('Calidad de Especialización')

    plt.tight_layout()
    plt.savefig('results/scalability_analysis.png', dpi=150)
    plt.show()

    return resultados
```
