from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from model.transformer import HyperbolicTransformer
from data.tokenizer import EnhancedTokenizer


class ErrorAnalyzer:
    """Analyze and categorize model errors"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Error categories
        self.error_categories = {
            'semantic': self._check_semantic_error,
            'structural': self._check_structural_error,
            'hierarchical': self._check_hierarchical_error,
            'attention': self._check_attention_error
        }
        
        # Error tracking
        self.error_history = defaultdict(list)
        
    def analyze_errors(self,
                      inputs: Dict[str, torch.Tensor],
                      outputs: Dict[str, torch.Tensor],
                      target: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive error analysis"""
        # Get predictions
        predictions = torch.argmax(outputs['logits'], dim=-1)
        
        # Compute error types
        error_types = {}
        for category, checker in self.error_categories.items():
            error_types[category] = checker(inputs, outputs, predictions, target)
            
        # Update error history
        self._update_error_history(error_types)
        
        # Generate error analysis
        analysis = {
            'error_types': error_types,
            'error_patterns': self._analyze_error_patterns(),
            'suggestions': self._generate_improvement_suggestions(error_types)
        }
        
        return analysis
    
    def _check_semantic_error(self,
                            inputs: Dict[str, torch.Tensor],
                            outputs: Dict[str, torch.Tensor],
                            predictions: torch.Tensor,
                            target: Optional[str]) -> Dict[str, float]:
        """Check for semantic-level errors"""
        semantic_errors = {
            'context_mismatch': self._compute_context_mismatch(outputs),
            'meaning_preservation': self._compute_meaning_preservation(
                inputs, outputs, predictions
            )
        }
        
        return semantic_errors
    
    def _check_structural_error(self,
                              inputs: Dict[str, torch.Tensor],
                              outputs: Dict[str, torch.Tensor],
                              predictions: torch.Tensor,
                              target: Optional[str]) -> Dict[str, float]:
        """Check for structural errors"""
        structural_errors = {
            'grammar': self._check_grammar_errors(predictions),
            'coherence': self._check_coherence_errors(outputs),
            'completeness': self._check_completeness_errors(outputs)
        }
        
        return structural_errors
    
    def _check_hierarchical_error(self,
                                inputs: Dict[str, torch.Tensor],
                                outputs: Dict[str, torch.Tensor],
                                predictions: torch.Tensor,
                                target: Optional[str]) -> Dict[str, float]:
        """Check for hierarchical representation errors"""
        hierarchical_errors = {
            'level_confusion': self._check_level_confusion(outputs),
            'hierarchy_preservation': self._check_hierarchy_preservation(outputs)
        }
        
        return hierarchical_errors
    
    def _check_attention_error(self,
                             inputs: Dict[str, torch.Tensor],
                             outputs: Dict[str, torch.Tensor],
                             predictions: torch.Tensor,
                             target: Optional[str]) -> Dict[str, float]:
        """Check for attention-related errors"""
        attention_errors = {
            'focus': self._check_attention_focus(outputs),
            'coverage': self._check_attention_coverage(outputs),
            'consistency': self._check_attention_consistency(outputs)
        }
        
        return attention_errors
    
    def _compute_context_mismatch(self,
                                outputs: Dict[str, torch.Tensor]) -> float:
        """Compute context mismatch score"""
        # Compare context vectors across layers
        context_vectors = torch.stack([
            layer_output['attention'].mean(dim=1)
            for layer_output in outputs['layer_outputs']
        ])
        
        # Compute consistency across layers
        consistency = F.cosine_similarity(
            context_vectors[:-1],
            context_vectors[1:],
            dim=-1
        ).mean()
        
        return 1 - consistency.item()
    
    def _compute_meaning_preservation(self,
                                   inputs: Dict[str, torch.Tensor],
                                   outputs: Dict[str, torch.Tensor],
                                   predictions: torch.Tensor) -> float:
        """Compute meaning preservation score"""
        # Get embeddings
        input_embeddings = self.model.token_embeddings(inputs['input_ids'])
        output_embeddings = outputs['last_hidden_state']
        
        # Compute semantic similarity
        similarity = F.cosine_similarity(
            input_embeddings.mean(dim=1),
            output_embeddings.mean(dim=1)
        )
        
        return similarity.item()
    
    def _update_error_history(self, error_types: Dict[str, Any]):
        """Update error history with new observations"""
        for category, errors in error_types.items():
            self.error_history[category].append(errors)
            
            # Keep history manageable
            if len(self.error_history[category]) > 1000:
                self.error_history[category].pop(0)
                
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in error history"""
        patterns = {}
        
        for category, history in self.error_history.items():
            patterns[category] = {
                'trend': self._compute_error_trend(history),
                'correlations': self._compute_error_correlations(history),
                'clusters': self._cluster_errors(history)
            }
            
        return patterns
    
    def _generate_improvement_suggestions(self,
                                       error_types: Dict[str, Any]) -> List[str]:
        """Generate suggestions for model improvement"""
        suggestions = []
        
        # Check different error categories
        if self._has_semantic_issues(error_types):
            suggestions.extend(self._get_semantic_suggestions())
            
        if self._has_structural_issues(error_types):
            suggestions.extend(self._get_structural_suggestions())
            
        if self._has_hierarchical_issues(error_types):
            suggestions.extend(self._get_hierarchical_suggestions())
            
        if self._has_attention_issues(error_types):
            suggestions.extend(self._get_attention_suggestions())
            
        return suggestions
    
    def _compute_error_trend(self, history: List[Any]) -> Dict[str, float]:
        """Compute trend in error metrics"""
        if not history:
            return {}
            
        # Convert to numpy for easier computation
        values = np.array([list(h.values()) for h in history])
        
        return {
            'mean': values.mean(axis=0).tolist(),
            'std': values.std(axis=0).tolist(),
            'slope': np.polyfit(
                np.arange(len(values)),
                values,
                deg=1
            )[0].tolist()
        }
    
    def _compute_error_correlations(self, history: List[Any]) -> np.ndarray:
        """Compute correlations between error metrics"""
        if not history:
            return np.array([])
            
        values = np.array([list(h.values()) for h in history])
        return np.corrcoef(values.T)
    
    def _cluster_errors(self, history: List[Any]) -> Dict[str, Any]:
        """Cluster error patterns"""
        if not history:
            return {}
            
        from sklearn.cluster import KMeans
        
        # Convert to feature matrix
        features = np.array([list(h.values()) for h in history])
        
        # Determine optimal number of clusters
        n_clusters = min(3, len(features))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features)
        
        return {
            'labels': clusters.tolist(),
            'centers': kmeans.cluster_centers_.tolist()
        }