import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularMoE(nn.Module):
    """Mixture of Experts para suma modular"""
    
    def __init__(self, input_dim=2, output_dim=1, num_experts=4, 
                 hidden_dim=64, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Dimensión de entrada (2 para a y b)
            output_dim: Dimensión de salida (1 para la suma)
            num_experts: Número de expertos
            hidden_dim: Dimensión de capas ocultas en cada experto
            num_layers: Número de capas ocultas por experto
            dropout: Dropout rate
        """
        super().__init__()
        self.num_experts = num_experts
        
        # Crear expertos (MLPs)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.experts.append(nn.Sequential(*layers))
        
        # Gate (enrutador) - decide qué expertos usar
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Inicialización
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, return_gate=True):
        """
        Args:
            x: Tensor de entrada (batch, input_dim)
            return_gate: Si devolver también los pesos del gate
        
        Returns:
            output: (batch, output_dim)
            gate_weights: (batch, num_experts) si return_gate=True
        """
        gate_weights = self.gate(x)  # (batch, num_experts)
        
        # Calcular salida de cada experto
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # cada uno: (batch, output_dim)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
        
        # Combinar según pesos del gate
        output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)  # (batch, output_dim)
        
        if return_gate:
            return output, gate_weights
        return output
    
    def get_expert_outputs(self, x):
        """Obtener salida de cada experto individualmente"""
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        return torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
    
    def get_routing_decision(self, x):
        """Obtener el experto con mayor peso para cada entrada"""
        with torch.no_grad():
            gate_weights = self.gate(x)
            return torch.argmax(gate_weights, dim=1)  # (batch,)