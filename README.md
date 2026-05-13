# Modular Addition with Mixture of Experts

[![DOI](https://zenodo.org/badge/1238198228.svg)](https://doi.org/10.5281/zenodo.20173358)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🧠 Descripción

Este repositorio implementa experimentos con arquitectura **Mixture of Experts (MoE)** para aprender operaciones de **suma modular** `(a + b) mod n`.

Inspirado en los experimentos de OpenAI sobre especialización de expertos, este proyecto demuestra que:

- ✅ Los expertos se especializan automáticamente en diferentes regiones del espacio
- ✅ **Módulos NO primos** (ej. 6) muestran mejor especialización que primos
- ✅ La estructura algebraica emerge sin supervisión explícita
- ✅ **60x más rápido** que implementaciones baseline

## 📊 Resultados Clave

### Especialización por módulo (4 expertos)

| Módulo | Tipo      | Entropía  | Expertos activos | Patrón                       |
| ------ | --------- | --------- | ---------------- | ---------------------------- |
| **6**  | No primo  | **0.323** | 4/4              | Franjas diagonales perfectas |
| **5**  | Primo     | 0.440     | 3/4              | Bloques 2×2 y 3×3            |
| **7**  | Primo     | 0.447     | 4/4              | Franjas con ruido            |
| **10** | Compuesto | 0.519     | 2/4              | Diagonal binaria             |

### Matriz de enrutamiento - Módulo 6 (mejor especialización)
