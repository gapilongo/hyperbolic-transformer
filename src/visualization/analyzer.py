import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer


class HyperbolicAnalyzer:
    """Analyze model behavior in hyperbolic space"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
    def analyze_embeddings(self,
                          dataloader: DataLoader) -> Dict[str, Any]:
        """Analyze embedding structure in hyperbolic space"""
        self.model.eval()
        embedding_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing embeddings"):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Get embeddings
                outputs = self.model(**batch)
                embeddings = outputs['last_hidden_state']
                
                # Compute statistics
                stats = self._analyze_hyperbolic_embeddings(embeddings)
                
                for key, value in stats.items():
                    embedding_stats[key].append(value)
                    
        # Average statistics
        final_stats = {}
        for key, values in embedding_stats.items():
            final_stats[key] = torch.stack(values).mean().item()
            
        return final_stats
    
    def _analyze_hyperbolic_embeddings(self,
                                     embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze properties of hyperbolic embeddings"""
        stats = {}
        
        # Compute distances from origin
        origin = torch.zeros_like(embeddings[:, 0])
        distances = self.model.hyperbolic.distance(embeddings, origin.unsqueeze(1))
        
        stats['mean_distance'] = distances.mean()
        stats['max_distance'] = distances.max()
        stats['distance_std'] = distances.std()
        
        # Compute pairwise distances
        pairwise_distances = []
        for i in range(embeddings.size(1)):
            for j in range(i + 1, embeddings.size(1)):
                dist = self.model.hyperbolic.distance(
                    embeddings[:, i],
                    embeddings[:, j]
                )
                pairwise_distances.append(dist)
                
        pairwise_distances = torch.stack(pairwise_distances, dim=0)
        
        stats['mean_pairwise_distance'] = pairwise_distances.mean()
        stats['max_pairwise_distance'] = pairwise_distances.max()
        stats['pairwise_distance_std'] = pairwise_distances.std()
        
        # Compute curvature utilization
        stats['curvature_utilization'] = self._compute_curvature_utilization(embeddings)
        
        return stats
    
    def _compute_curvature_utilization(self,
                                     embeddings: torch.Tensor) -> torch.Tensor:
        """Compute how well the model utilizes hyperbolic curvature"""
        # Project embeddings to tangent space at origin
        origin = torch.zeros_like(embeddings[:, 0])
        log_map = self.model.hyperbolic.log_map(origin.unsqueeze(1), embeddings)
        
        # Compute ratio of hyperbolic to Euclidean distances
        hyperbolic_distances = self.model.hyperbolic.distance(embeddings, origin.unsqueeze(1))
        euclidean_distances = torch.norm(log_map, dim=-1)
        
        # Ratio > 1 indicates utilization of hyperbolic curvature
        distance_ratio = hyperbolic_distances / (euclidean_distances + 1e-8)
        
        return distance_ratio.mean()

