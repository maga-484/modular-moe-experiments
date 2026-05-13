import torch
import torch.nn as nn

class SharedExpert(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class SpecializedHead(nn.Module):
    def __init__(self, in_features=512, out_features=5):
        super().__init__()
        self.head = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.head(x)

class SharedExpertMoE(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=5):
        super().__init__()
        self.shared = SharedExpert(input_dim, hidden_dim, output_dim)
        self.heads = nn.ModuleList()
        self.emb_dim = hidden_dim
        self.out_dim = output_dim

    def add_head(self):
        new_head = SpecializedHead(self.emb_dim, self.out_dim)
        self.heads.append(new_head)
        return len(self.heads) - 1

    def forward(self, x, silo_ids):
        """
        x: tensor de features (batch_size, input_dim)
        silo_ids: tensor de enteros (batch_size,) con el índice de la cabeza a usar para cada muestra
        """
        features = self.shared(x)  # (batch, hidden_dim)
        # Recolectar salidas de cada cabeza según silo_id
        outputs = []
        for i, sid in enumerate(silo_ids):
            outputs.append(self.heads[sid](features[i]))
        return torch.stack(outputs)  # (batch, output_dim)

    def freeze_shared(self):
        for p in self.shared.parameters():
            p.requires_grad = False

    def unfreeze_shared(self):
        for p in self.shared.parameters():
            p.requires_grad = True