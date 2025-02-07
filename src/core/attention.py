import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from src.core.config.configurations import ModelConfig
from src.core.hyperbolic import HyperbolicSpace
import math

class HyperbolicGraphAttention(nn.Module):
    """Graph attention mechanism in hyperbolic space"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Attention projections
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Hyperbolic space
        self.hyperbolic = HyperbolicSpace(dim=config.hidden_size)
        
        # Edge importance prediction
        self.edge_importance = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hyperbolic attention
        
        Args:
            node_embeddings: [batch_size, num_nodes, hidden_size]
            attention_mask: Optional [batch_size, num_nodes]
            adjacency_matrix: Optional [batch_size, num_nodes, num_nodes]
            
        Returns:
            updated_embeddings: [batch_size, num_nodes, hidden_size]
            attention_weights: [batch_size, num_heads, num_nodes, num_nodes]
        """
        batch_size, num_nodes = node_embeddings.size()[:2]
        device = node_embeddings.device
        
        # Project queries, keys, values
        queries = self.query_proj(node_embeddings).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        keys = self.key_proj(node_embeddings).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        values = self.value_proj(node_embeddings).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute hyperbolic attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale
        
        # Create default adjacency matrix if not provided
        if adjacency_matrix is None:
            # Allow all connections
            adjacency_matrix = torch.ones(batch_size, num_nodes, num_nodes, device=device)
        
        # Apply adjacency mask
        adjacency_mask = (adjacency_matrix == 0).unsqueeze(1)
        scores = scores.masked_fill(adjacency_mask, -1e9)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + attention_mask
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, values)
        
        # Merge heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.hidden_size
        )
        
        # Project output
        output = self.output_proj(context)
        
        # Project back to hyperbolic space
        output = self.hyperbolic.exp_map(
            node_embeddings,
            self.hyperbolic.log_map(node_embeddings, output)
        )
        
        return output, attention_weights