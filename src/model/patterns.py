import torch.nn as nn
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from src.core.config.configurations import ModelConfig
from src.core.hyperbolic import HyperbolicSpace
from src.core.tensor import TensorManager


class DimensionHandler:
    """Framework's tensor dimension handling system"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.shape_registry = {}
        
    def validate_shape(self,
                      tensor: torch.Tensor,
                      expected_shape: Tuple[Optional[int], ...],
                      operation: str) -> bool:
        """Validate tensor shape against expected shape pattern"""
        if len(tensor.shape) != len(expected_shape):
            return False
            
        for actual, expected in zip(tensor.shape, expected_shape):
            if expected is not None and actual != expected:
                return False
                
        return True
        
    def prepare_attention_input(self, 
                              tensor: torch.Tensor,
                              num_heads: Optional[int] = None) -> torch.Tensor:
        """Prepare tensor for attention operations"""
        original_shape = tensor.shape
        self.shape_registry[id(tensor)] = original_shape
        
        if len(original_shape) == 4:  # [batch, heads, seq, dim]
            return tensor.reshape(-1, original_shape[2], original_shape[3])
        elif len(original_shape) == 3:  # [batch, seq, dim]
            if num_heads:
                return tensor.reshape(
                    original_shape[0],
                    original_shape[1],
                    num_heads,
                    -1
                ).transpose(1, 2)
        return tensor

    def prepare_hyperbolic_input(self,
                                tensor: torch.Tensor) -> torch.Tensor:
        """Prepare tensor for hyperbolic operations"""
        original_shape = tensor.shape
        self.shape_registry[id(tensor)] = original_shape
        
        if len(original_shape) == 4:  # Handle pattern dimension
            return tensor.view(-1, original_shape[-1])
        elif len(original_shape) == 3:  # Standard case
            return tensor.view(-1, original_shape[-1])
        return tensor
        
    def prepare_pattern_input(self,
                            tensor: torch.Tensor,
                            pattern_size: Optional[int] = None) -> torch.Tensor:
        """Prepare tensor for pattern operations"""
        original_shape = tensor.shape
        self.shape_registry[id(tensor)] = original_shape
        
        if pattern_size is None:
            pattern_size = self.config.pattern_size
            
        if len(original_shape) == 3:  # [batch, seq, dim]
            return tensor.view(-1, pattern_size, original_shape[1], original_shape[2])
        return tensor

    def restore_shape(self,
                     tensor: torch.Tensor,
                     tensor_id: Optional[int] = None,
                     target_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Restore tensor to original or target shape"""
        if tensor_id is not None and tensor_id in self.shape_registry:
            original_shape = self.shape_registry[tensor_id]
        elif target_shape is not None:
            original_shape = target_shape
        else:
            raise ValueError("Must provide either tensor_id or target_shape")
            
        return tensor.view(original_shape)

    def get_attention_shape(self,
                          batch_size: int,
                          seq_length: int,
                          num_heads: Optional[int] = None) -> torch.Size:
        """Get expected shape for attention operations"""
        if num_heads is None:
            num_heads = self.config.num_attention_heads
            
        return torch.Size([batch_size, num_heads, seq_length, self.config.hidden_size // num_heads])

    def get_pattern_shape(self,
                         batch_size: int,
                         pattern_size: Optional[int] = None) -> torch.Size:
        """Get expected shape for pattern operations"""
        if pattern_size is None:
            pattern_size = self.config.pattern_size
            
        return torch.Size([batch_size, pattern_size, self.config.hidden_size])
    

class PatternLearner(nn.Module):
    """Advanced pattern learning in hyperbolic space"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tensor_manager = TensorManager(config)
        self.hyperbolic = HyperbolicSpace(dim=config.hidden_size)
        
        # Enhanced pattern networks in hyperbolic space
        self.pattern_encoder = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        ])
        
        self.pattern_decoder = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        ])
        
        # Hyperbolic pattern memory
        self.num_patterns = config.num_patterns
        self.pattern_memory = nn.Parameter(
            self.hyperbolic.exp_map(
                torch.zeros(self.num_patterns, config.hidden_size),
                torch.randn(self.num_patterns, config.hidden_size) * 0.02
            )
        )
        self.pattern_importance = nn.Parameter(torch.ones(self.num_patterns))
        
    def encode_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into hyperbolic pattern space"""
        # Map to tangent space at origin
        x_tangent = self.hyperbolic.log_map(
            torch.zeros_like(x),
            x
        )
        
        # Apply encoding in tangent space
        for layer in self.pattern_encoder:
            x_tangent = layer(x_tangent)
            
        # Map back to hyperbolic space
        return self.hyperbolic.exp_map(
            torch.zeros_like(x_tangent),
            x_tangent
        )
        
    def decode_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Decode pattern from hyperbolic space"""
        # Map to tangent space
        pattern_tangent = self.hyperbolic.log_map(
            torch.zeros_like(pattern),
            pattern
        )
        
        # Apply decoding in tangent space
        for layer in self.pattern_decoder:
            pattern_tangent = layer(pattern_tangent)
            
        # Map back to hyperbolic space
        return self.hyperbolic.exp_map(
            torch.zeros_like(pattern_tangent),
            pattern_tangent
        )
        
    def find_similar_patterns(self, 
                            pattern: torch.Tensor,
                            top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find similar patterns in hyperbolic memory"""
        if top_k is None:
            top_k = self.config.pattern_top_k
            
        # Compute hyperbolic distances
        distances = []
        for stored_pattern, importance in zip(self.pattern_memory, self.pattern_importance):
            # Compute distance in hyperbolic space
            distance = self.hyperbolic.distance(pattern, stored_pattern)
            # Apply importance weighting
            weighted_distance = distance * torch.sigmoid(importance)
            distances.append(weighted_distance)
            
        distances = torch.stack(distances)
        
        # Get top-k closest patterns
        top_distances, indices = torch.topk(distances, k=min(top_k, len(distances)), largest=False)
        top_patterns = self.pattern_memory[indices]
        
        return top_patterns, torch.exp(-top_distances)  # Convert distances to similarities
        
    def update_pattern_memory(self, 
                            new_pattern: torch.Tensor,
                            similarity_threshold: float = 0.8) -> None:
        """Update pattern memory in hyperbolic space"""
        # Find most similar existing pattern
        similarities = []
        for stored_pattern in self.pattern_memory:
            similarity = torch.exp(-self.hyperbolic.distance(new_pattern, stored_pattern))
            similarities.append(similarity)
            
        similarities = torch.stack(similarities)
        max_similarity, max_idx = torch.max(similarities, dim=0)
        
        if max_similarity > similarity_threshold:
            # Update existing pattern using parallel transport
            update_vector = self.hyperbolic.log_map(
                self.pattern_memory[max_idx],
                new_pattern
            )
            
            # Apply gradual update
            lr = self.config.pattern_learning_rate
            update_vector = update_vector * lr
            
            self.pattern_memory.data[max_idx] = self.hyperbolic.exp_map(
                self.pattern_memory.data[max_idx],
                update_vector
            )
            
            # Update importance
            self.pattern_importance.data[max_idx] *= 1.1
        else:
            # Replace least important pattern
            min_importance, min_idx = torch.min(self.pattern_importance, dim=0)
            self.pattern_memory.data[min_idx] = new_pattern
            self.pattern_importance.data[min_idx] = 1.0
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with explicit tensor shape management"""
        # Get and validate input dimensions
        batch_size, seq_len, hidden_size = x.shape
        
        # Ensure dimensions match configuration
        if hidden_size != self.config.hidden_size:
            raise ValueError(f"Hidden size mismatch. Expected {self.config.hidden_size}, got {hidden_size}")
        
        # Process input in batches to avoid memory issues
        pattern = self.encode_pattern(x)  # [batch_size, seq_len, hidden_size]
        
        # Find similar patterns - reshape pattern properly for comparison
        flattened_pattern = pattern.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        all_patterns = []
        all_similarities = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 128  # Can be configured in ModelConfig
        for i in range(0, flattened_pattern.size(0), chunk_size):
            chunk = flattened_pattern[i:i + chunk_size]
            chunk_patterns, chunk_similarities = self.find_similar_patterns(chunk)
            all_patterns.append(chunk_patterns)
            all_similarities.append(chunk_similarities)
        
        # Combine results
        similar_patterns = torch.cat(all_patterns, dim=0)  # [(batch_size * seq_len), num_patterns, hidden_size]
        similarities = torch.cat(all_similarities, dim=0)   # [(batch_size * seq_len), num_patterns]
        
        # Process patterns maintaining proper dimensions
        processed = []
        for i in range(batch_size):
            for j in range(seq_len):
                idx = i * seq_len + j
                current_pattern = pattern[i, j].unsqueeze(0)  # [1, hidden_size]
                current_similar = similar_patterns[idx]        # [num_patterns, hidden_size]
                current_weights = similarities[idx]           # [num_patterns]
                
                # Weighted combination
                weighted_sum = (current_similar * current_weights.unsqueeze(-1)).sum(dim=0)
                processed.append(weighted_sum)
        
        # Reshape back to original dimensions
        processed = torch.stack(processed).view(batch_size, seq_len, hidden_size)
        
        # Decode processed patterns
        output = self.decode_pattern(processed)
        
        return output, pattern
    
class HierarchicalPatternProcessor:
    """Process and organize patterns hierarchically"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.pattern_learner = PatternLearner(config)
        self.pattern_hierarchy = defaultdict(list)
        self.level_patterns = defaultdict(dict)
        
    def process_patterns(self, 
                        embeddings: torch.Tensor,
                        levels: List[int]) -> Dict[int, torch.Tensor]:
        """Process embeddings at different hierarchical levels"""
        results = {}
        
        for level in sorted(set(levels)):
            # Get embeddings at this level
            level_mask = torch.tensor([l == level for l in levels])
            level_embeddings = embeddings[level_mask]
            
            if len(level_embeddings) == 0:
                continue
                
            # Process patterns at this level
            processed, patterns = self.pattern_learner(level_embeddings)
            
            # Store patterns
            self.level_patterns[level].update({
                i: p for i, p in enumerate(patterns)
            })
            
            # Update hierarchy
            for pattern in patterns:
                self.pattern_hierarchy[level].append(pattern)
                
            results[level] = processed
            
        return results
    
    def find_cross_level_patterns(self,
                                query_pattern: torch.Tensor,
                                source_level: int,
                                target_level: int,
                                top_k: int = 5) -> List[torch.Tensor]:
        """Find related patterns across hierarchical levels"""
        if target_level not in self.level_patterns:
            return []
            
        target_patterns = list(self.level_patterns[target_level].values())
        if not target_patterns:
            return []
            
        # Stack patterns for efficient computation
        target_stack = torch.stack(target_patterns)
        
        # Compute similarities
        similarities = []
        for pattern in target_stack:
            similarity = -self.pattern_learner.hyperbolic.distance(
                query_pattern,
                pattern
            )
            similarities.append(similarity)
            
        similarities = torch.stack(similarities)
        
        # Get top-k patterns
        top_similarities, indices = torch.topk(
            similarities,
            k=min(top_k, len(similarities))
        )
        
        return [target_patterns[i] for i in indices]