import torch
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import torch.nn.functional as F
from core.config.configurations import ModelConfig


class PatternMemory:
    """Memory for storing and predicting patterns"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.patterns = []
        self.pattern_embeddings = []
        
        # Pattern matching parameters
        self.max_patterns = 1000
        self.similarity_threshold = 0.8
        self.min_pattern_length = 3
        
    def add_pattern(self, pattern: torch.Tensor, embedding: torch.Tensor):
        """Add new pattern to memory"""
        if len(self.patterns) >= self.max_patterns:
            # Remove oldest pattern
            self.patterns.pop(0)
            self.pattern_embeddings.pop(0)
            
        self.patterns.append(pattern)
        self.pattern_embeddings.append(embedding)
        
    def predict_next(self,
                   current_embedding: torch.Tensor,
                   generated_patterns: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Predict next token based on patterns"""
        if not self.patterns:
            return None
            
        # Find similar patterns
        similarities = []
        for pattern_embedding in self.pattern_embeddings:
            similarity = F.cosine_similarity(
                current_embedding,
                pattern_embedding,
                dim=-1
            )
            similarities.append(similarity)
            
        similarities = torch.stack(similarities)
        
        # Get most similar pattern
        max_similarity, max_idx = torch.max(similarities, dim=0)
        
        if max_similarity > self.similarity_threshold:
            pattern = self.patterns[max_idx]
            return pattern
            
        return None

class LRUCache:
    """Least Recently Used Cache"""
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache"""
        if key not in self.cache:
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key: Any, value: Any):
        """Add item to cache"""
        if key in self.cache:
            # Move to end
            self.cache.move_to_end(key)
        else:
            # Check size
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                self.cache.popitem(last=False)
                
        self.cache[key] = value