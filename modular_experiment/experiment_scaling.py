# modular_experiment/experiment_scaling.py
def test_scaling(moduli=[5,6,7,8,9,10,11,12,13,14,15]):
    results = []
    
    for n in moduli:
        print(f"\n📊 Probando módulo {n}...")
        
        # Entrenar
        model = train(modulo=n, num_experts=min(8, n), epochs=200)
        
        # Evaluar
        routing, _ = analyze_routing(model, max_num=n-1, modulo=n)
        entropy = compute_entropy(routing)
        active = len(np.unique(routing))
        
        results.append({
            'modulo': n,
            'entropy': entropy,
            'active_experts': active,
            'time': elapsed_time
        })
        
        # Guardar checkpoint
        save_results(results)
    
    # Visualizar escalabilidad
    plot_scalability(results)
    return results