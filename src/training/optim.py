import torch
from torch.optim import AdamW
import math
from src.model.transformer import HyperbolicTransformer
from src.core.config.configurations import TrainingConfig


class HyperbolicOptimizer:
    """Custom optimizer for hyperbolic space"""
    def __init__(self, 
                 model: HyperbolicTransformer,
                 config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Split parameters into hyperbolic and euclidean
        hyperbolic_params = []
        euclidean_params = []
        
        for name, param in model.named_parameters():
            # Checking for components that should use hyperbolic optimization
            if any(x in name for x in [
                'hyperbolic',
                'graph',
                'attention_layers.edge_importance'
            ]):
                hyperbolic_params.append(param)
            else:
                euclidean_params.append(param)
                
        # Debug info
        print(f"\nParameter split:")
        print(f"Euclidean parameters: {len(euclidean_params)}")
        print(f"Hyperbolic parameters: {len(hyperbolic_params)}")
        
        # Ensure we have parameters for both optimizers
        if not euclidean_params:
            raise ValueError("No euclidean parameters found!")
        if not hyperbolic_params:
            # If no hyperbolic parameters, use all parameters for both optimizers
            print("\nNo explicit hyperbolic parameters found. Using all parameters for both optimizers.")
            hyperbolic_params = list(model.parameters())
                
        # Create optimizers
        self.euclidean_optimizer = AdamW(
            euclidean_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.hyperbolic_optimizer = RiemannianAdam(
            hyperbolic_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def step(self):
        """Take an optimization step"""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Update parameters
        self.euclidean_optimizer.step()
        self.hyperbolic_optimizer.step()
        
    def zero_grad(self):
        """Zero all gradients"""
        self.euclidean_optimizer.zero_grad()
        self.hyperbolic_optimizer.zero_grad()

class RiemannianAdam(torch.optim.Optimizer):
    """Adam optimizer adapted for Riemannian manifolds"""
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        # Validate params
        if not isinstance(params, (list, tuple)):
            params = list(params)
        if not params:
            raise ValueError("RiemannianAdam received empty parameter list")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step in hyperbolic space"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get parameter state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                # Update step count
                state['step'] += 1
                
                # Get hyperparameters
                beta1, beta2 = group['betas']
                
                # Update momentum and variance
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p
                    
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Get bias-corrected estimates
                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step'])
                bias_correction1 = 1 - beta1 ** state['step']
                
                # Compute update direction
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = exp_avg / bias_correction1 / denom
                
                # Project update to tangent space
                if hasattr(p, 'manifold'):
                    update = p.manifold.proj(update, p)
                    
                # Apply update
                p.add_(update, alpha=-step_size)
                
                # Project back to manifold if needed
                if hasattr(p, 'manifold'):
                    p.data.copy_(p.manifold.proj(p.data))