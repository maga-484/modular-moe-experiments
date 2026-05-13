import torch

def generate_silo_data(seed, n_samples=500, input_dim=32, output_dim=32, variation=0.05):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    torch.manual_seed(42)
    W_base = torch.randn(input_dim, output_dim) * 0.8
    torch.manual_seed(seed)
    W_var = torch.randn(input_dim, output_dim) * variation
    W = W_base + W_var
    y = torch.mm(X, W) + torch.randn(n_samples, output_dim) * 0.05
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True) + 1e-8
    y = (y - y_mean) / y_std
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - X_mean) / X_std
    return X, y
