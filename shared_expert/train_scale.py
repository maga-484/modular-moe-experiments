import torch
import torch.optim as optim
import time
from .model import SharedExpertMoE
from .generator import generate_silo_data
from .utils import get_memory_usage

def train_shared_expert(num_silos=100, epochs=50, lr=0.001, verbose=False):
    model = SharedExpertMoE()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    silos_data = []
    for i in range(num_silos):
        variation = 0.05 + (i / num_silos) * 0.45
        X, y = generate_silo_data(seed=1000+i, variation=variation)
        silos_data.append((X, y))
        model.add_head()

    start_time = time.time()
    ram_start, vram_start = get_memory_usage()

    for epoch in range(epochs):
        total_loss = 0.0
        for head_id, (X, y) in enumerate(silos_data):
            optimizer.zero_grad()
            pred = model(X, head_id)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_silos
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    end_time = time.time()
    ram_end, vram_end = get_memory_usage()

    timediff = end_time - start_time
    ram_used = ram_end - ram_start
    vram_used = vram_end - vram_start

    print(f"\n✅ Entrenamiento de {num_silos} silos completado.")
    print(f"   Tiempo: {timediff:.2f} segundos")
    print(f"   Memoria RAM: {ram_used:.1f} MB")
    if vram_used > 0:
        print(f"   Memoria VRAM: {vram_used:.1f} MB")
    return model, silos_data, (timediff, ram_used, vram_used)
