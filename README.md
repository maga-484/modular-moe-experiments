# Modular Addition Mixture of Experts (MoE)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🧠 Descripción

Implementación de una arquitectura **Mixture of Experts (Shared Expert MoE)** para aprender operaciones de suma modular, replicando y extendiendo los experimentos de OpenAI sobre especialización de expertos.

### Descubrimientos clave

- ✅ **Especialización automática**: Los expertos aprenden diferentes regiones del espacio de suma modular
- ✅ **Módulos no primos son más fáciles**: Contraintuitivamente, ℤ/6ℤ muestra mejor especialización que ℤ/5ℤ o ℤ/7ℤ
- ✅ **Estructura algebraica emergente**: La red descubre subgrupos y cosets automáticamente
- ✅ **60x más rápido** que implementaciones baseline

## 📊 Resultados Experimentales

### Entropía de enrutamiento (menor = mejor especialización)

| Módulo | Tipo | Entropía | Expertos activos | Patrón |
|--------|------|----------|------------------|--------|
| 6 | No primo | **0.323** | 4/4 | Franjas diagonales |
| 5 | Primo | 0.440 | 3/4 | Bloques 2×2, 3×3 |
| 7 | Primo | 0.447 | 4/4 | Franjas + ruido |
| 10 | Compuesto | 0.519 | 2/4 | Diagonal binaria |

### Matrices de enrutamiento

**Módulo 5 (4 expertos):**