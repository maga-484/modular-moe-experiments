import subprocess
import sys

def run_experiment(script_name):
    print(f"\n{'='*80}")
    print(f"EJECUTANDO: {script_name}")
    print(f"{'='*80}")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    return result.returncode

if __name__ == "__main__":
    experiments = [
        ("modular_experiment/experiment_mod10.py", "Experimento 1: Módulo 10"),
        ("modular_experiment/experiment_8experts.py", "Experimento 2: 8 Expertos"),
        ("modular_experiment/experiment_visualize_functions.py", "Experimento 3: Visualización"),
        ("modular_experiment/experiment_prime_vs_nonprime.py", "Experimento 4: Primo vs No primo"),
    ]
    
    for script, name in experiments:
        input(f"\nPresiona Enter para comenzar {name}...")
        run_experiment(script)
    
    print("\n✅ Todos los experimentos completados!")