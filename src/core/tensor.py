import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Any
from src.core.config.configurations import ModelConfig
import math

class TensorManager:
    """Framework's core tensor management system"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.supported_ops = {
            'attention': self._attention_reshape,
            'pattern': self._pattern_reshape,
            'hyperbolic': self._hyperbolic_reshape,
            'graph': self._graph_reshape
        }
    def _track_shape(self, tensor_id: int, shape: torch.Size, operation: str):
        if self.shape_history is not None:
            self.shape_history[tensor_id] = {
                'shape': shape,
                'operation': operation
            }

    def verify_shape_consistency(self, tensor: torch.Tensor, expected_shape: torch.Size) -> bool:
        if not hasattr(self.config, 'verify_shapes') or not self.config.verify_shapes:
            return True
        return tensor.shape == expected_shape


    def validate_dimensions(self, 
                        tensor: torch.Tensor, 
                        operation: str) -> Tuple[bool, str]:
        """Framework dimension validation with detailed feedback"""
        expected_dims = {
            'attention': self.config.input_dim,
            'pattern': (3, 4),  # Accept both 3D and 4D for patterns
            'hyperbolic': 3,
            'graph': 3
        }
        
        if operation not in expected_dims:
            return False, f"Unsupported operation: {operation}"
            
        actual_dims = len(tensor.shape)
        expected = expected_dims[operation]
        
        # Handle tuple case for multiple acceptable dimensions
        if isinstance(expected, tuple):
            is_valid = actual_dims in expected
            message = (f"Expected one of {expected} dimensions for {operation}, "
                    f"got {actual_dims}. Shape: {tensor.shape}")
        else:
            is_valid = actual_dims == expected
            message = (f"Expected {expected} dimensions for {operation}, "
                    f"got {actual_dims}. Shape: {tensor.shape}")
                    
        return is_valid, message

    def reshape_tensor(self, 
                      tensor: torch.Tensor,
                      operation: str,
                      **kwargs) -> torch.Tensor:
        """Framework standard tensor reshaping"""
        if operation not in self.supported_ops:
            raise ValueError(f"Unsupported operation: {operation}")
            
        is_valid, message = self.validate_dimensions(tensor, operation)
        if not is_valid and not kwargs.get('force', False):
            raise ValueError(message)
            
        return self.supported_ops[operation](tensor, **kwargs)

    def _attention_reshape(self, 
                         tensor: torch.Tensor,
                         num_heads: Optional[int] = None) -> torch.Tensor:
        """Framework standard attention reshape"""
        if num_heads is None:
            num_heads = self.config.num_attention_heads
            
        if len(tensor.shape) == 4:  # [batch, heads, seq, dim]
            return tensor.transpose(1, 2).reshape(
                tensor.size(0),
                tensor.size(2),
                -1
            )
        elif len(tensor.shape) == 3:  # [batch, seq, dim]
            return tensor.reshape(
                tensor.size(0),
                tensor.size(1),
                num_heads,
                -1
            ).transpose(1, 2)
        else:
            return tensor.unsqueeze(0)

    def _pattern_reshape(self, 
                        tensor: torch.Tensor,
                        pattern_size: Optional[int] = None) -> torch.Tensor:
        """Framework standard pattern reshape"""
        if pattern_size is None:
            pattern_size = getattr(self.config, 'pattern_size', 32)  # Default size if not configured
            
        # Handle 3D input case [batch, seq, dim]
        if len(tensor.shape) == 3:
            batch_size, seq_len, dim = tensor.shape
            # Reshape to [batch, pattern_size, seq/pattern_size, dim]
            num_segments = max(1, seq_len // pattern_size)
            return tensor.view(batch_size, num_segments, pattern_size, dim)
            
        # Handle 4D input case [batch, patterns, seq, dim]
        elif len(tensor.shape) == 4:
            return tensor
            
        else:
            raise ValueError(f"Unsupported tensor shape for pattern reshape: {tensor.shape}")

    def _hyperbolic_reshape(self, 
                           tensor: torch.Tensor,
                           **kwargs) -> torch.Tensor:
        """Framework standard hyperbolic reshape"""
        return tensor.view(tensor.size(0), -1, self.config.hidden_size)

    def _graph_reshape(self, 
                      tensor: torch.Tensor,
                      **kwargs) -> torch.Tensor:
        """Framework standard graph reshape"""
        return tensor.view(tensor.size(0), -1, self.config.hidden_size)

    def restore_dimensions(self, 
                         tensor: torch.Tensor,
                         original_shape: torch.Size) -> torch.Tensor:
        """Restore tensor to original dimensions"""
        return tensor.view(original_shape)

class EnhancedTensorNetwork(nn.Module):
    """Advanced tensor network with comprehensive dimension handling"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tensor_manager = TensorManager(config)
        
        # Core components with proper initialization
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize network components with proper scaling"""
        hidden_size = self.config.hidden_size
        self.head_dim = hidden_size // self.config.num_attention_heads
        
        # Core tensors with proper initialization
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Pattern components
        self.pattern_projection = nn.Linear(hidden_size, self.config.tensor_bond_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Framework standard weight initialization"""
        for module in [self.query, self.key, self.value, self.output, self.pattern_projection]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Enhanced forward pass with comprehensive dimension handling"""
        # Process input
        original_shape = hidden_states.shape
        has_patterns = len(original_shape) == 4
        
        if has_patterns:
            hidden_states = self.tensor_manager.reshape_tensor(
                hidden_states,
                'pattern'
            )
            
        # Core attention operations with proper reshaping
        query = self.tensor_manager.reshape_tensor(
            self.query(hidden_states),
            'attention'
        )
        key = self.tensor_manager.reshape_tensor(
            self.key(hidden_states),
            'attention'
        )
        value = self.tensor_manager.reshape_tensor(
            self.value(hidden_states),
            'attention'
        )
        
        # Compute attention with proper scaling
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context and output
        context = torch.matmul(attention_probs, value)
        output = self.output(
            self.tensor_manager.restore_dimensions(context, original_shape)
        )
        
        if return_attention:
            return output, attention_probs
        return output