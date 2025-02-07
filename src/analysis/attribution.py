from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import itertools
import torch.nn.functional as F
from model.transformer import HyperbolicTransformer
from data.tokenizer import EnhancedTokenizer



class AttributionComputer:
    """Compute and analyze feature attributions"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Attribution methods
        self.methods = {
            'integrated_gradients': self._compute_integrated_gradients,
            'attention': self._compute_attention_attribution,
            'hyperbolic': self._compute_hyperbolic_attribution
        }
        
    def compute_attributions(self,
                           inputs: Dict[str, torch.Tensor],
                           outputs: Dict[str, torch.Tensor],
                           method: str = 'integrated_gradients') -> Dict[str, torch.Tensor]:
        """Compute attributions using specified method"""
        if method not in self.methods:
            raise ValueError(f"Unknown attribution method: {method}")
            
        # Compute base attributions
        attributions = self.methods[method](inputs, outputs)
        
        # Add hyperbolic context
        attributions = self._add_hyperbolic_context(attributions, outputs)
        
        return attributions
    
    def _compute_integrated_gradients(self,
                                    inputs: Dict[str, torch.Tensor],
                                    outputs: Dict[str, torch.Tensor],
                                    steps: int = 50) -> torch.Tensor:
        """Compute Integrated Gradients attributions"""
        # Get input embeddings
        embeddings = self.model.token_embeddings(inputs['input_ids'])
        
        # Create baseline (zero embeddings)
        baseline = torch.zeros_like(embeddings)
        
        # Compute integral
        attributions = torch.zeros_like(embeddings)
        for alpha in torch.linspace(0, 1, steps):
            interpolated = baseline + alpha * (embeddings - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(inputs_embeds=interpolated)
            logits = outputs['logits']
            
            # Compute gradients
            gradients = torch.autograd.grad(
                logits.sum(),
                interpolated
            )[0]
            
            attributions += gradients
            
        attributions *= (embeddings - baseline) / steps
        
        return attributions
    
    def _compute_attention_attribution(self,
                                    inputs: Dict[str, torch.Tensor],
                                    outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute attributions based on attention weights"""
        # Get attention weights from all layers
        attentions = torch.stack([
            layer_output['attention'].mean(dim=1)  # Average across heads
            for layer_output in outputs['layer_outputs']
        ])
        
        # Combine attention across layers
        combined_attention = attentions.mean(dim=0)  # Average across layers
        
        return combined_attention
    
    def _compute_hyperbolic_attribution(self,
                                     inputs: Dict[str, torch.Tensor],
                                     outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute attributions in hyperbolic space"""
        # Get hyperbolic embeddings
        embeddings = outputs['last_hidden_state']
        
        # Compute distances from decision boundary
        logits = outputs['logits']
        
        # Project logits into hyperbolic space
        hyperbolic_logits = self.model.hyperbolic.exp_map(
            torch.zeros_like(logits),
            logits
        )
        
        # Compute distances to decision boundary
        distances = self.model.hyperbolic.distance(
            embeddings,
            hyperbolic_logits
        )
        
        # Normalize distances to attributions
        attributions = F.softmax(-distances, dim=-1)
        
        return attributions
    
    def _add_hyperbolic_context(self,
                               attributions: torch.Tensor,
                               outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add hyperbolic geometric context to attributions"""
        # Get hyperbolic embeddings
        embeddings = outputs['last_hidden_state']
        
        # Compute geometric features
        curvature = self._compute_local_curvature(embeddings)
        distances = self._compute_pairwise_distances(embeddings)
        hierarchy = self._compute_hierarchical_position(embeddings)
        
        return {
            'raw_attributions': attributions,
            'curvature_context': curvature,
            'distance_context': distances,
            'hierarchy_context': hierarchy
        }
    
    def _compute_local_curvature(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute local curvature around each embedding"""
        # Get neighboring points
        neighbors = self._get_nearest_neighbors(embeddings)
        
        # Compute sectional curvature
        curvature = torch.zeros(embeddings.size(0))
        for i in range(embeddings.size(0)):
            if len(neighbors[i]) >= 2:
                # Take pairs of neighbors
                for j, k in itertools.combinations(neighbors[i], 2):
                    # Compute triangle area
                    area = self._hyperbolic_triangle_area(
                        embeddings[i],
                        embeddings[j],
                        embeddings[k]
                    )
                    curvature[i] += area
                    
        return curvature
    
    def _compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances"""
        n = embeddings.size(0)
        distances = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                distances[i,j] = self.model.hyperbolic.distance(
                    embeddings[i],
                    embeddings[j]
                )
                
        return distances
    
    def _compute_hierarchical_position(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical position in hyperbolic space"""
        # Compute distance from origin
        origin = torch.zeros_like(embeddings[0])
        distances = self.model.hyperbolic.distance(embeddings, origin)
        
        # Normalize to [0,1] range
        hierarchy = F.softmax(-distances, dim=-1)
        
        return hierarchy