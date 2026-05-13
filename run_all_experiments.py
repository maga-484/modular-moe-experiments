import subprocess
import sys
import os

def run_experiment(script_name, description):
    print(f"\n{'='*80}")
    print(f"EJECUTANDO: {description}")
    print(f"{'='*80}")
    
    # Verificar si el script existe
    if not os.path.exists(script_name):
        print(f"⚠️  Error: No se encuentra {script_name}")
        print(f"   Asegurate de que el archivo existe en la ruta correcta")
        return 1
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    return result.returncode

if __name__ == "__main__":
    print("="*80)
    print("🚀 INICIANDO EXPERIMENTOS COMPLETOS DE MoE PARA SUMA MODULAR")
    print("="*80)
    
    # Lista de experimentos (ruta del script, descripción)
    experiments = [
        ("modular_experiment/experiment_mod10.py", "Experimento 1: Módulo 10 (0-9)"),
        ("modular_experiment/experiment_8experts.py", "Experimento 2: 8 Expertos"),
        ("modular_experiment/experiment_visualize_functions.py", "Experimento 3: Visualización de funciones"),
        ("modular_experiment/experiment_prime_vs_nonprime.py", "Experimento 4: Primo (7) vs No primo (6)"),
    ]
    
    # Ejecutar cada experimento
    for script, description in experiments:
        input(f"\n✨ Presiona ENTER para comenzar: {description}")
        return_code = run_experiment(script, description)
        
        if return_code != 0:
            print(f"❌ Error en {description}. Código: {return_code}")
            respuesta = input("¿Continuar con el siguiente experimento? (s/n): ")
            if respuesta.lower() != 's':
                break
        else:
            print(f"✅ {description} completado exitosamente!")
    
    print("\n" + "="*80)
    print("🎉 TODOS LOS EXPERIMENTOS COMPLETADOS!")
    print("📁 Resultados guardados en carpeta 'results/'")
    print("="*80)