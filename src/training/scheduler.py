import torch
import numpy as np

class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int = 1000,
                 max_steps: int = 100000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Initial learning rate
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # Tracking
        self.step_num = 0
        self.history = []
        
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        lr = self._compute_lr()
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.history.append(lr)
        
    def _compute_lr(self) -> float:
        """Compute current learning rate"""
        if self.step_num < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.step_num / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_num - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
            
    def get_last_lr(self) -> float:
        """Get current learning rate"""
        return self.history[-1] if self.history else self.initial_lr