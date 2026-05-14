import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKMoE(nn.Module):
    """
    Top-K Mixture of Experts - Arquitectura estándar en LLMs modernos
    (Mixtral, DeepSeek, etc.)
    
    Solo activa los K mejores expertos por muestra.
    """
    
    def __init__(self, input_dim=2, output_dim=1, num_experts=4, 
                 top_k=2, hidden_dim=64, num_layers=2, dropout=0.1,
                 noise=False, noise_std=0.1):
        """
        Args:
            input_dim: Dimensión de entrada (2 para a y b)
            output_dim: Dimensión de salida (1 para la suma)
            num_experts: Número total de expertos
            top_k: Número de expertos a activar por muestra (típico 2)
            hidden_dim: Dimensión de capas ocultas
            num_layers: Número de capas por experto
            dropout: Dropout rate
            noise: Si añadir ruido Gumbel para exploración
            noise_std: Desviación estándar del ruido
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise = noise
        self.noise_std = noise_std
        
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
        
        # Gate (enrutador) - produce logits para cada experto
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, return_gate=False):
        """
        Forward pass con Top-K routing.
        
        Args:
            x: Tensor de entrada (batch, input_dim)
            return_gate: Si devolver información del gate
        
        Returns:
            output: (batch, output_dim)
            gate_info: Dict con información del routing
        """
        batch_size = x.shape[0]
        
        # Calcular logits del gate
        gate_logits = self.gate(x)  # (batch, num_experts)
        
        # Añadir ruido Gumbel para exploración (si se entrena)
        if self.training and self.noise:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # Normalizar entre los K
        
        # Calcular salidas de expertos (solo los seleccionados)
        expert_outputs = []
        for i in range(batch_size):
            expert_outputs_i = []
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                expert_out = self.experts[expert_idx](x[i:i+1])
                expert_outputs_i.append(expert_out * top_k_weights[i, k])
            expert_outputs.append(torch.sum(torch.stack(expert_outputs_i), dim=0))
        
        output = torch.cat(expert_outputs, dim=0)  # (batch, output_dim)
        
        if return_gate:
            gate_info = {
                'logits': gate_logits,
                'top_k_indices': top_k_indices,
                'top_k_weights': top_k_weights,
                'expert_usage': self._get_expert_usage(top_k_indices)
            }
            return output, gate_info
        return output
    
    def _get_expert_usage(self, top_k_indices):
        """Calcular cuántas veces se usa cada experto en el batch"""
        usage = torch.zeros(self.num_experts)
        for idx in top_k_indices.flatten():
            usage[idx] += 1
        return usage / len(top_k_indices.flatten())
    
    def get_routing_decision(self, x):
        """Obtener los K mejores expertos para cada entrada"""
        with torch.no_grad():
            gate_logits = self.gate(x)
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        return top_k_indices
    
    def get_expert_outputs(self, x):
        """Obtener salida de cada experto individualmente"""
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        return torch.stack(expert_outputs, dim=1)


class TopKMoEAnalyzer:
    """Analizador de routing para Top-k MoE"""
    
    @staticmethod
    def compute_routing_matrix(model, max_num=4, modulo=None, device='cpu'):
        """Calcular matriz de enrutamiento (experto dominante)"""
        routing_matrix = torch.zeros((max_num + 1, max_num + 1), dtype=torch.long)
        
        with torch.no_grad():
            for a in range(max_num + 1):
                for b in range(max_num + 1):
                    x = torch.tensor([[a, b]], dtype=torch.float32).to(device)
                    # Para top-k, usamos el experto con mayor peso
                    top_k_indices = model.get_routing_decision(x)
                    routing_matrix[a, b] = top_k_indices[0, 0]  # Primer experto (mayor peso)
        
        return routing_matrix
    
    @staticmethod
    def compute_expert_balance(model, dataloader, device='cpu'):
        """Calcular balance de uso de expertos en un dataset"""
        expert_counts = torch.zeros(model.num_experts)
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                _, gate_info = model(x, return_gate=True)
                expert_counts += gate_info['expert_usage'] * len(x)
                total_samples += len(x)
        
        return expert_counts / total_samples
    
    @staticmethod
    def compute_load_balancing_loss(gate_logits, top_k_indices, top_k_weights):
        """
        Calcular pérdida de balance de carga (auxiliary loss)
        Para evitar que todos se concentren en pocos expertos.
        """
        num_experts = gate_logits.shape[-1]
        
        # Fracción de muestras enrutadas a cada experto
        f_i = torch.zeros(num_experts, device=gate_logits.device)
        for i in range(num_experts):
            f_i[i] = (top_k_indices == i).float().mean()
        
        # Fracción de peso asignado a cada experto
        p_i = torch.zeros(num_experts, device=gate_logits.device)
        for i in range(num_experts):
            mask = (top_k_indices == i)
            if mask.any():
                p_i[i] = (top_k_weights * mask.float()).sum() / mask.sum()
        
        # Coeficiente de variación (menor = más balanceado)
        load_balance_loss = torch.var(f_i + p_i) * num_experts
        return load_balance_loss