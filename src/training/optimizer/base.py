import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from typing import Dict, List, Optional, Any, Callable
from core.config.configurations import ModelConfig

class TrainingOptimizer:
    """Optimize training process dynamically"""
    def __init__(self,
                 model: torch.nn.Module,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
        # Initialize optimizers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        
        # Training state tracking
        self.step_count = 0
        self.loss_history = []
        self.grad_history = []
        
    def _setup_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Setup multiple optimizers for different components"""
        optimizers = {
            'main': torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        }
        
        # Add specialized optimizers for different components
        if hasattr(self.model, 'hyperbolic'):
            optimizers['hyperbolic'] = torch.optim.RiemannianAdam(
                self.model.hyperbolic.parameters(),
                lr=self.config.learning_rate
            )
            
        if hasattr(self.model, 'attention'):
            optimizers['attention'] = torch.optim.Adam(
                self.model.attention.parameters(),
                lr=self.config.learning_rate * 0.1  # Lower LR for attention
            )
            
        return optimizers
    
    def _setup_schedulers(self) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate schedulers"""
        schedulers = {}
        
        # Main scheduler with OneCycle policy
        schedulers['main'] = OneCycleLR(
            self.optimizers['main'],
            max_lr=self.config.learning_rate,
            total_steps=self.config.total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Cosine scheduling with restarts for specialized optimizers
        for name in ['hyperbolic', 'attention']:
            if name in self.optimizers:
                schedulers[name] = CosineAnnealingWarmRestarts(
                    self.optimizers[name],
                    T_0=self.config.epochs // 4,
                    T_mult=2
                )
                
        return schedulers
    
    def optimization_step(self,
                         loss: torch.Tensor,
                         update_schedulers: bool = True):
        """Perform optimized training step"""
        # Gradient scaling for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Compute gradients
        scaler.scale(loss).backward()
        
        # Track gradients
        grad_norm = self._compute_grad_norm()
        self.grad_history.append(grad_norm)
        
        # Optimize each component
        for name, optimizer in self.optimizers.items():
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self._get_param_group(name),
                self.config.max_grad_norm
            )
            
            # Update parameters
            scaler.step(optimizer)
            
            # Update scheduler
            if update_schedulers and name in self.schedulers:
                self.schedulers[name].step()
                
        # Update scaler
        scaler.update()
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
            
        self.step_count += 1
        
    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def _get_param_group(self, name: str) -> List[torch.nn.Parameter]:
        """Get parameter group for specific optimizer"""
        if name == 'hyperbolic':
            return self.model.hyperbolic.parameters()
        elif name == 'attention':
            return self.model.attention.parameters()
        else:
            return self.model.parameters()
    
    def get_current_lr(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {
            name: scheduler.get_last_lr()[0]
            for name, scheduler in self.schedulers.items()
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'step_count': self.step_count,
            'learning_rates': self.get_current_lr(),
            'grad_norm_mean': np.mean(self.grad_history[-100:]),
            'grad_norm_std': np.std(self.grad_history[-100:])
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing"""
        return {
            'optimizers': {
                name: opt.state_dict()
                for name, opt in self.optimizers.items()
            },
            'schedulers': {
                name: sched.state_dict()
                for name, sched in self.schedulers.items()
            },
            'step_count': self.step_count,
            'grad_history': self.grad_history
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint"""
        for name, opt_state in state_dict['optimizers'].items():
            self.optimizers[name].load_state_dict(opt_state)
            
        for name, sched_state in state_dict['schedulers'].items():
            self.schedulers[name].load_state_dict(sched_state)
            
        self.step_count = state_dict['step_count']
        self.grad_history = state_dict['grad_history']