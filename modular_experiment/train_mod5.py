from modular_experiment.train import train

if __name__ == "__main__":
    train(modulo=5, num_experts=4, epochs=200, hidden_dim=64)