import logging
from collections import defaultdict
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from src.core.config.configurations import ModelConfig
from src.model.transformer import HyperbolicTransformer

class DebugMonitor:
    """Monitor and debug model behavior"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Gradient tracking
        self.grad_norms = defaultdict(list)
        self.grad_flows = defaultdict(list)
        
        # Activation tracking
        self.activation_stats = defaultdict(list)
        self.dead_neurons = set()
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        for name, module in self.model.named_modules():
            module.register_forward_hook(
                lambda m, i, o, name=name: self._forward_hook(m, i, o, name)
            )
            module.register_backward_hook(
                lambda m, i, o, name=name: self._backward_hook(m, i, o, name)
            )
            
    def _forward_hook(self,
                     module: torch.nn.Module,
                     inputs: tuple,
                     output: torch.Tensor,
                     name: str):
        """Track activations in forward pass"""
        if isinstance(output, torch.Tensor):
            stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'zero_fraction': (output == 0).float().mean().item()
            }
            
            self.activation_stats[name].append(stats)
            
            # Check for dead neurons
            if stats['zero_fraction'] > 0.99:
                self.dead_neurons.add(name)
                
    def _backward_hook(self,
                      module: torch.nn.Module,
                      grad_input: tuple,
                      grad_output: tuple,
                      name: str):
        """Track gradients in backward pass"""
        if grad_output[0] is not None:
            self.grad_norms[name].append(
                grad_output[0].norm().item()
            )
            
            self.grad_flows[name].append(
                grad_output[0].mean().item()
            )
            
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        return {
            'gradient_health': self._analyze_gradients(),
            'activation_health': self._analyze_activations(),
            'dead_neurons': list(self.dead_neurons),
            'recommendations': self._generate_recommendations()
        }
        
    def _analyze_gradients(self) -> Dict[str, Any]:
        """Analyze gradient behavior"""
        gradient_health = {}
        
        for name, norms in self.grad_norms.items():
            gradient_health[name] = {
                'mean_norm': np.mean(norms),
                'norm_std': np.std(norms),
                'vanishing': np.mean(norms) < 1e-4,
                'exploding': np.mean(norms) > 1e2,
                'flow_stability': np.std(self.grad_flows[name]) / (np.mean(self.grad_flows[name]) + 1e-8)
            }
            
        return gradient_health
    
    def _analyze_activations(self) -> Dict[str, Any]:
        """Analyze activation behavior"""
        activation_health = {}
        
        for name, stats_list in self.activation_stats.items():
            means = [s['mean'] for s in stats_list]
            stds = [s['std'] for s in stats_list]
            zero_fracs = [s['zero_fraction'] for s in stats_list]
            
            activation_health[name] = {
                'mean_stability': np.std(means) / (np.mean(means) + 1e-8),
                'std_stability': np.std(stds) / (np.mean(stds) + 1e-8),
                'saturation': np.mean([s['max'] > 0.99 or s['min'] < -0.99 for s in stats_list]),
                'sparsity': np.mean(zero_fracs)
            }
            
        return activation_health
    
    def _generate_recommendations(self) -> List[str]:
        """Generate debugging recommendations"""
        recommendations = []
        
        # Check gradient issues
        grad_health = self._analyze_gradients()
        for name, health in grad_health.items():
            if health['vanishing']:
                recommendations.append(
                    f"Vanishing gradients detected in {name}. "
                    "Consider using skip connections or gradient scaling."
                )
            elif health['exploding']:
                recommendations.append(
                    f"Exploding gradients detected in {name}. "
                    "Consider using gradient clipping or layer normalization."
                )
                
        # Check activation issues
        act_health = self._analyze_activations()
        for name, health in act_health.items():
            if health['saturation'] > 0.5:
                recommendations.append(
                    f"High activation saturation in {name}. "
                    "Consider reducing learning rate or using batch normalization."
                )
            if health['sparsity'] > 0.9:
                recommendations.append(
                    f"High activation sparsity in {name}. "
                    "Consider checking initialization or reducing regularization."
                )
                
        return recommendations
