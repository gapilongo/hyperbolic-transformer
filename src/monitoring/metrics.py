from typing import List, Dict, Optional, Union, Any
import torch
import torch.nn.functional as F
from collections import defaultdict
from src.core.config.configurations import ModelConfig

class LossComputer:
    """Advanced loss computation with multiple objectives"""
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Loss weights
        self.mlm_weight = 1.0
        self.nsp_weight = 0.5
        self.structure_weight = 0.3
        self.pattern_weight = 0.2
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor],
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all loss components"""
        total_loss = 0.0
        loss_dict = {}
        
        # Masked Language Modeling loss
        if 'mlm_labels' in batch:
            mlm_loss = self.compute_mlm_loss(
                outputs['logits'],
                batch['mlm_labels']
            )
            total_loss += self.mlm_weight * mlm_loss
            loss_dict['mlm_loss'] = mlm_loss
            
        # Next Sentence Prediction loss
        if 'nsp_label' in batch:
            nsp_loss = self.compute_nsp_loss(
                outputs['nsp_logits'],
                batch['nsp_label']
            )
            total_loss += self.nsp_weight * nsp_loss
            loss_dict['nsp_loss'] = nsp_loss
            
        # Structure preservation loss
        structure_loss = self.compute_structure_loss(outputs)
        total_loss += self.structure_weight * structure_loss
        loss_dict['structure_loss'] = structure_loss
        
        # Pattern consistency loss
        pattern_loss = self.compute_pattern_loss(outputs)
        total_loss += self.pattern_weight * pattern_loss
        loss_dict['pattern_loss'] = pattern_loss
        
        loss_dict['total_loss'] = total_loss
        
        # Update history
        for key, value in loss_dict.items():
            self.loss_history[key].append(value.item())
            
        return loss_dict
    
    def compute_mlm_loss(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """Compute masked language modeling loss"""
        # Only compute loss on masked tokens
        active_loss = labels != -100
        active_logits = logits[active_loss]
        active_labels = labels[active_loss]
        
        loss = F.cross_entropy(
            active_logits,
            active_labels,
            reduction='mean'
        )
        
        return loss
    
    def compute_nsp_loss(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """Compute next sentence prediction loss"""
        return F.cross_entropy(logits, labels, reduction='mean')
    
    def compute_structure_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute structure preservation loss"""
        # Get hyperbolic embeddings
        embeddings = outputs['last_hidden_state']
        
        # Compute pairwise distances in hyperbolic space
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Get attention weights from last layer
        attention_weights = outputs['layer_outputs'][-1]['attention']
        attention_weights = attention_weights.mean(dim=1)  # Average over heads
        
        # Structure preservation loss
        loss = F.mse_loss(
            dist_matrix,
            -torch.log(attention_weights + 1e-10)
        )
        
        return loss
    
    def compute_pattern_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute pattern consistency loss"""
        patterns = outputs['layer_outputs'][-1]['patterns']
        
        # Get pattern similarities across layers
        pattern_similarities = []
        for i in range(len(outputs['layer_outputs']) - 1):
            curr_patterns = outputs['layer_outputs'][i]['patterns']
            next_patterns = outputs['layer_outputs'][i + 1]['patterns']
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                curr_patterns.view(-1, curr_patterns.size(-1)),
                next_patterns.view(-1, next_patterns.size(-1))
            )
            pattern_similarities.append(similarity)
            
        # Encourage pattern consistency across layers
        pattern_similarities = torch.stack(pattern_similarities)
        loss = -torch.mean(pattern_similarities)
        
        return loss

class MetricsComputer:
    """Compute and track various metrics"""
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def compute_metrics(self,
                       outputs: Dict[str, torch.Tensor],
                       batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute all relevant metrics"""
        metrics = {}
        
        # MLM accuracy
        if 'mlm_labels' in batch:
            metrics['mlm_accuracy'] = self.compute_mlm_accuracy(
                outputs['logits'],
                batch['mlm_labels']
            )
            
        # NSP accuracy
        if 'nsp_label' in batch:
            metrics['nsp_accuracy'] = self.compute_nsp_accuracy(
                outputs['nsp_logits'],
                batch['nsp_label']
            )
            
        # Perplexity
        metrics['perplexity'] = self.compute_perplexity(
            outputs['logits'],
            batch['mlm_labels']
        )
        
        # Update history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
        return metrics
    
    def compute_mlm_accuracy(self,
                           logits: torch.Tensor,
                           labels: torch.Tensor) -> float:
        """Compute masked language modeling accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        active_accuracy = labels != -100
        
        correct = (predictions[active_accuracy] == labels[active_accuracy]).float().mean()
        return correct.item()
    
    def compute_nsp_accuracy(self,
                           logits: torch.Tensor,
                           labels: torch.Tensor) -> float:
        """Compute next sentence prediction accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float().mean()
        return correct.item()
    
    def compute_perplexity(self,
                          logits: torch.Tensor,
                          labels: torch.Tensor) -> float:
        """Compute perplexity"""
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        return torch.exp(loss).item()