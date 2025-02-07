from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer

class PatternAnalyzer:
    """Analyze learned patterns in model behavior"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
        # Pattern tracking
        self.pattern_history = defaultdict(list)
        self.pattern_clusters = {}
        
    def analyze_patterns(self,
                        outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze patterns in model outputs"""
        # Extract relevant patterns
        patterns = {
            'attention': self._analyze_attention_patterns(outputs),
            'embedding': self._analyze_embedding_patterns(outputs),
            'hyperbolic': self._analyze_hyperbolic_patterns(outputs)
        }
        
        # Update pattern history
        self._update_pattern_history(patterns)
        
        # Cluster patterns
        self._cluster_patterns()
        
        return {
            'current_patterns': patterns,
            'pattern_clusters': self.pattern_clusters,
            'recurring_patterns': self._get_recurring_patterns()
        }
    
    def _analyze_attention_patterns(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Analyze attention pattern characteristics"""
        attention_patterns = {}
        
        for layer_idx, layer_output in enumerate(outputs['layer_outputs']):
            attention = layer_output['attention']  # [batch_size, num_heads, seq_length, seq_length]
            
            # Reshape to 3D for analysis
            batch_size, num_heads, seq_length, _ = attention.size()
            attention_3d = attention.view(batch_size * num_heads, seq_length, seq_length)
            
            attention_patterns[f'layer_{layer_idx}'] = {
                'concentration': self._compute_attention_concentration(attention_3d),
                'consistency': self._compute_attention_consistency(attention_3d),
                'structure': self._compute_attention_structure(attention_3d)
            }
                
        return attention_patterns
    
    def _analyze_embedding_patterns(self,
                                 outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Analyze embedding space patterns"""
        embeddings = outputs['last_hidden_state']
        
        # Compute embedding characteristics
        embedding_patterns = {
            'clustering': self._analyze_embedding_clusters(embeddings),
            'trajectory': self._analyze_embedding_trajectory(embeddings)
        }
        
        return embedding_patterns
    
    def _analyze_hyperbolic_patterns(self,
                                   outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Analyze patterns in hyperbolic space"""
        # Get hyperbolic embeddings
        embeddings = outputs['last_hidden_state']
        
        # Analyze hyperbolic characteristics
        hyperbolic_patterns = {
            'curvature': self._analyze_curvature_patterns(embeddings),
            'hierarchy': self._analyze_hierarchical_patterns(embeddings)
        }
        
        return hyperbolic_patterns
    
    def _compute_attention_concentration(self,
                                      attention: torch.Tensor) -> torch.Tensor:
        """Compute attention concentration metrics"""
        # Compute Gini coefficient of attention weights
        sorted_attention = torch.sort(attention, dim=-1)[0]
        n = attention.size(-1)
        indices = torch.arange(1, n + 1, device=attention.device)
        gini = 2 * (indices * sorted_attention).sum(dim=-1) / (n * sorted_attention.sum(dim=-1)) - (n + 1) / n
        return gini.mean()
    
    def _compute_attention_consistency(self,
                                    attention: torch.Tensor) -> torch.Tensor:
        """Compute consistency of attention patterns"""
        # Compare attention patterns across heads
        mean_attention = attention.mean(dim=1, keepdim=True)
        consistency = F.cosine_similarity(attention, mean_attention, dim=-1).mean()
        return consistency
    
    def _compute_attention_structure(self,
                                  attention: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze structural properties of attention"""
        # Compute various structural metrics
        return {
            'symmetry': self._compute_attention_symmetry(attention),
            'locality': self._compute_attention_locality(attention),
            'hierarchy': self._compute_attention_hierarchy(attention)
        }
        
    def _update_pattern_history(self, patterns: Dict[str, Any]):
        """Update pattern history with new observations"""
        for pattern_type, pattern_data in patterns.items():
            self.pattern_history[pattern_type].append(pattern_data)
            
            # Keep history size manageable
            if len(self.pattern_history[pattern_type]) > 1000:
                self.pattern_history[pattern_type].pop(0)
                
    def _cluster_patterns(self):
        """Cluster observed patterns"""
        from sklearn.cluster import DBSCAN
        
        for pattern_type, history in self.pattern_history.items():
            if len(history) < 10:  # Need minimum samples for clustering
                continue
                
            # Convert patterns to feature vectors
            features = self._patterns_to_features(history)
            
            # Cluster patterns
            clustering = DBSCAN(eps=0.3, min_samples=5)
            clusters = clustering.fit_predict(features)
            
            # Store cluster information
            self.pattern_clusters[pattern_type] = {
                'labels': clusters,
                'centers': self._compute_cluster_centers(features, clusters)
            }
            
    def _patterns_to_features(self, patterns: List[Any]) -> np.ndarray:
        """Convert patterns to feature vectors for clustering"""
        # Implementation depends on pattern type
        # This is a placeholder for the actual feature extraction logic
        return np.array([self._extract_features(p) for p in patterns])
    
    def _extract_features(self, pattern: Any) -> np.ndarray:
        """Extract numerical features from a pattern"""
        # Implementation depends on pattern type
        # This is a placeholder for the actual feature extraction logic
        return np.array([0.0])  # Placeholder
    
    def _get_recurring_patterns(self) -> Dict[str, List[Any]]:
        """Identify recurring patterns in history"""
        recurring_patterns = {}
        
        for pattern_type, clusters in self.pattern_clusters.items():
            # Find patterns that appear frequently
            unique_clusters, counts = np.unique(clusters['labels'], return_counts=True)
            frequent_clusters = unique_clusters[counts > len(self.pattern_history[pattern_type]) * 0.1]
            
            recurring_patterns[pattern_type] = [
                {
                    'cluster_id': cluster_id,
                    'frequency': count / len(self.pattern_history[pattern_type]),
                    'center': clusters['centers'][cluster_id]
                }
                for cluster_id, count in zip(unique_clusters, counts)
                if cluster_id in frequent_clusters
            ]
            
        return recurring_patterns