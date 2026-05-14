# Modular Addition with Mixture of Experts

DOI: 10.5281/zenodo.20173358

[![DOI](https://zenodo.org/badge/1238198228.svg)](https://doi.org/10.5281/zenodo.20173358)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Descripción

Este repositorio implementa experimentos completos con arquitectura **Mixture of Experts (MoE)** para aprender operaciones aritméticas modulares:

- **Suma modular** `(a + b) mod n`
- **Multiplicación modular** `(a × b) mod n`
- **Potencia modular** `(a^b) mod n`
- **Polinomio cuadrático** `(a² + b) mod n`

El proyecto investiga cómo los expertos se especializan espontáneamente en diferentes regiones del espacio algebraico.

## 🏆 Resultados Completos

### Ranking Absoluto - Módulo 5

| Puesto | Operación      | Arquitectura | Mejor Loss   | Hallazgo                 |
| ------ | -------------- | ------------ | ------------ | ------------------------ |
| 🥇     | **Potencia**   | Full MoE     | **0.000040** | Récord absoluto          |
| 🥈     | Multiplicación | Full MoE     | 0.000195     | Muy bueno                |
| 🥉     | Potencia       | Top-2 MoE    | 0.000196     | Top-2 no mejora potencia |
| 4      | Suma           | Top-2 MoE    | 0.000412     | Top-2 mejora suma (3.3x) |
| 5      | Polinomio      | Full MoE     | 0.000832     | Más difícil              |
| 6      | Suma           | Full MoE     | 0.00134      | Línea base               |

### Ranking - Módulo 10

| Puesto | Operación      | Arquitectura | Mejor Loss | ¿Aprende?  |
| ------ | -------------- | ------------ | ---------- | ---------- |
| 🥇     | Multiplicación | Full MoE     | 0.000801   | ✅ Sí      |
| 🥈     | Potencia       | Full MoE     | 0.000918   | ✅ Sí      |
| 🥉     | Suma           | Full MoE     | 0.000954   | ✅ Sí      |
| 4      | Suma           | Top-2 MoE    | 0.003126   | 🟡 Regular |
| 5      | Polinomio      | Full MoE     | 0.0420     | ❌ No      |
| 6      | Potencia       | Top-2 MoE    | 0.0987     | ❌ No      |

## 🔬 Descubrimientos Clave

### 1. La complejidad no predice la dificultad

- **Potencia** (exponencial) es más fácil que **suma** (lineal)
- La estructura de grupo cíclico facilita el aprendizaje

### 2. Top-2 MoE: útil solo para suma

- Suma módulo 5: **3.3x mejor** que Full MoE
- Potencia módulo 10: **100x peor** que Full MoE
- La arquitectura debe elegirse según la operación

### 3. El polinomio es el límite del MoE

- En módulo 10: **52x peor** que multiplicación
- La no-linealidad pura confunde al router

### 4. Load Loss como diagnóstico

- Load Loss > 0.5 → señal temprana de fallo
- Valores bajos (<0.1) indican buen balance

## 📊 Comparativa de Operaciones

| Operación       | Módulo 5 | Módulo 10 | Dificultad      |
| --------------- | -------- | --------- | --------------- |
| Potencia (Full) | 0.000040 | 0.000918  | Muy fácil       |
| Multiplicación  | 0.000195 | 0.000801  | Fácil           |
| Suma (Top-2)    | 0.000412 | 0.003126  | Media           |
| Polinomio       | 0.000832 | 0.0420    | **Muy difícil** |

## 📈 Visualizaciones

El repositorio incluye:

- `power_embeddings_mod5.png` - Estructura de grupo cíclico
- `poly_embeddings_mod5.png` - Embeddings del polinomio
- `final_comparison_all_operations.png` - Comparativa global
- Matrices de enrutamiento para todas las operaciones

## 🚀 Instalación y Uso

```bash
# Clonar repositorio
git clone https://github.com/maga-484/modular-moe-experiments.git
cd modular-moe-experiments

# Instalar dependencias
pip install -r requirements.txt

# Entrenar suma modular
python -m modular_experiment.train --modulo 5 --epochs 200

# Entrenar multiplicación modular
python -m modular_experiment.train_multiplication --modulo 5 --epochs 200

# Entrenar potencia modular
python -m modular_experiment.train_power --modulo 5 --epochs 200

 Estructura del Proyecto
text
modular-moe-experiments/
├── modular_experiment/
│   ├── dataset.py          # Suma modular
│   ├── dataset_mul.py      # Multiplicación
│   ├── dataset_pow.py      # Potencia
│   ├── dataset_poly.py     # Polinomios
│   ├── model.py            # Full MoE
│   ├── model_topk.py       # Top-2 MoE
│   ├── train.py            # Entrenamiento suma
│   ├── train_multiplication.py
│   ├── train_power.py
│   ├── train_power_topk.py
│   ├── analyze_power_embeddings.py
│   └── analyze_poly.py
├── results/                # Visualizaciones
├── models_shared/          # Modelos entrenados
└── README.md
📚 Referencias
OpenAI "Outsider Learning" (2022)

Shared Expert Mixture of Experts (Ma et al., 2018)
```
