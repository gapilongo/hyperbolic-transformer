import torch
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import deque, defaultdict
from src.core.config.configurations import AdaptiveConfig, ModelConfig
from src.training.optimizer.base import TrainingOptimizer

class AdaptiveOptimizer:
    """Dynamic optimization with adaptive policies"""
    def __init__(self,
                 training_optimizer: TrainingOptimizer,
                 config: AdaptiveConfig):
        self.optimizer = training_optimizer
        self.config = config
        
        # State tracking
        self.loss_window = deque(maxlen=config.window_size)
        self.grad_window = deque(maxlen=config.window_size)
        self.steps_since_adjust = 0
        
        # Policy states
        self.current_policy = self._initialize_policy()
        self.policy_history = []
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
    def _initialize_policy(self) -> Dict[str, Any]:
        """Initialize adaptive policy parameters"""
        return {
            'learning_rate': self.optimizer.config.learning_rate,
            'batch_size': self.optimizer.config.batch_size,
            'momentum': 0.9,
            'gradient_clip': self.optimizer.config.max_grad_norm
        }
        
    def step(self, loss: float, grad_norm: float):
        """Update adaptive policies"""
        # Track metrics
        self.loss_window.append(loss)
        self.grad_window.append(grad_norm)
        self.steps_since_adjust += 1
        
        # Check if adaptation is needed
        if self._should_adapt():
            self._adapt_policy()
            self.steps_since_adjust = 0
            
    def _should_adapt(self) -> bool:
        """Determine if policy adaptation is needed"""
        if len(self.loss_window) < self.config.window_size:
            return False
            
        if self.steps_since_adjust < self.config.min_steps_per_adjust:
            return False
            
        # Check loss trend
        loss_trend = self._compute_loss_trend()
        if loss_trend > self.config.target_loss_change:
            return True
            
        # Check gradient health
        if self._has_gradient_issues():
            return True
            
        return False
    
    def _compute_loss_trend(self) -> float:
        """Compute trend in loss values"""
        if len(self.loss_window) < 2:
            return 0.0
            
        # Compute relative change over window
        window_list = list(self.loss_window)
        return (np.mean(window_list[-10:]) - np.mean(window_list[:10])) / np.mean(window_list[:10])
    
    def _has_gradient_issues(self) -> bool:
        """Check for gradient-related issues"""
        if len(self.grad_window) < self.config.window_size:
            return False
            
        grad_norms = np.array(list(self.grad_window))
        
        # Check for vanishing gradients
        if np.mean(grad_norms) < 1e-4:
            return True
            
        # Check for exploding gradients
        if np.mean(grad_norms) > 100:
            return True
            
        # Check for high variance
        if np.std(grad_norms) / np.mean(grad_norms) > 2.0:
            return True
            
        return False
    
    def _adapt_policy(self):
        """Adapt optimization policy based on performance"""
        # Store current policy
        self.policy_history.append(self.current_policy.copy())
        
        # Analyze recent performance
        loss_trend = self._compute_loss_trend()
        grad_issues = self._has_gradient_issues()
        
        # Update learning rate
        self._adapt_learning_rate(loss_trend, grad_issues)
        
        # Update batch size
        self._adapt_batch_size(loss_trend)
        
        # Update momentum
        self._adapt_momentum(grad_issues)
        
        # Update gradient clipping
        self._adapt_gradient_clip(grad_issues)
        
        # Apply new policy
        self._apply_policy()
        
    def _adapt_learning_rate(self, loss_trend: float, grad_issues: bool):
        """Adapt learning rate based on performance"""
        current_lr = self.current_policy['learning_rate']
        
        if grad_issues:
            # Reduce learning rate for gradient issues
            new_lr = current_lr * 0.5
        elif loss_trend > 0:
            # Loss increasing, reduce learning rate
            new_lr = current_lr * 0.7
        elif loss_trend > self.config.target_loss_change:
            # Loss decreasing too slowly
            new_lr = current_lr * 1.3
        else:
            # Loss decreasing well, minor adjustment
            new_lr = current_lr * (1.0 + 0.1 * np.sign(self.config.target_loss_change - loss_trend))
            
        # Ensure within bounds
        self.current_policy['learning_rate'] = np.clip(
            new_lr,
            self.config.lr_bounds[0],
            self.config.lr_bounds[1]
        )
        
    def _adapt_batch_size(self, loss_trend: float):
        """Adapt batch size based on performance"""
        current_bs = self.current_policy['batch_size']
        
        if loss_trend > 0:
            # Loss increasing, reduce batch size
            new_bs = max(current_bs // 2, self.config.batch_size_bounds[0])
        elif self._check_gpu_utilization() < 0.7:
            # GPU underutilized, increase batch size
            new_bs = min(current_bs * 2, self.config.batch_size_bounds[1])
        else:
            # Keep current batch size
            new_bs = current_bs
            
        self.current_policy['batch_size'] = new_bs
        
    def _adapt_momentum(self, grad_issues: bool):
        """Adapt momentum based on gradient behavior"""
        current_momentum = self.current_policy['momentum']
        
        if grad_issues:
            # Reduce momentum for stability
            new_momentum = max(current_momentum * 0.9, self.config.momentum_bounds[0])
        else:
            # Gradually increase momentum
            new_momentum = min(current_momentum * 1.1, self.config.momentum_bounds[1])
            
        self.current_policy['momentum'] = new_momentum
        
    def _adapt_gradient_clip(self, grad_issues: bool):
        """Adapt gradient clipping threshold"""
        current_clip = self.current_policy['gradient_clip']
        
        if grad_issues:
            # Tighten gradient clipping
            self.current_policy['gradient_clip'] = current_clip * 0.8
        else:
            # Gradually relax clipping
            self.current_policy['gradient_clip'] = current_clip * 1.2
            
    def _apply_policy(self):
        """Apply updated policy to optimizer"""
        # Update learning rates
        for name, scheduler in self.optimizer.schedulers.items():
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.max_lr = self.current_policy['learning_rate']
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.base_lr = self.current_policy['learning_rate']
                
        # Update momentum
        for optimizer in self.optimizer.optimizers.values():
            if isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
                for param_group in optimizer.param_groups:
                    if 'momentum' in param_group:
                        param_group['momentum'] = self.current_policy['momentum']
                    elif 'betas' in param_group:
                        param_group['betas'] = (
                            self.current_policy['momentum'],
                            param_group['betas'][1]
                        )
                        
        # Update gradient clipping
        self.optimizer.config.max_grad_norm = self.current_policy['gradient_clip']
        
    def _check_gpu_utilization(self) -> float:
        """Check GPU utilization"""
        if torch.cuda.is_available():
            return torch.cuda.utilization() / 100.0
        return 0.0
        
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        return {
            'current_policy': self.current_policy,
            'steps_since_adjust': self.steps_since_adjust,
            'loss_trend': self._compute_loss_trend(),
            'grad_health': not self._has_gradient_issues(),
            'policy_changes': len(self.policy_history)
        }
    
class AdaptiveTrainer:
    """Training manager with adaptive optimization"""
    def __init__(self,
                 model: torch.nn.Module,
                 config: ModelConfig,
                 adaptive_config: AdaptiveConfig):
        self.model = model
        self.config = config
        self.adaptive_config = adaptive_config
        
        # Initialize components
        self.training_optimizer = TrainingOptimizer(model, config)
        self.adaptive_optimizer = AdaptiveOptimizer(
            self.training_optimizer,
            adaptive_config
        )
        
        # State tracking
        self.step_count = 0
        self.epoch_count = 0
        self.best_loss = float('inf')
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step with adaptation"""
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['loss']
        
        # Optimization step
        self.training_optimizer.optimization_step(loss)
        
        # Compute gradient norm
        grad_norm = self.training_optimizer._compute_grad_norm()
        
        # Update adaptive policies
        self.adaptive_optimizer.step(loss.item(), grad_norm)
        
        # Update counters
        self.step_count += 1
        
        # Track best loss
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            
        # Return metrics
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            **self.adaptive_optimizer.get_adaptation_stats()
        }
        
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with adaptation"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Record metrics
            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)
                
        # Compute epoch statistics
        epoch_stats = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        self.epoch_count += 1
        
        return epoch_stats
    
    def save_checkpoint(self, path: str):
        """Save training state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.training_optimizer.state_dict(),
            'adaptive_policy': self.adaptive_optimizer.current_policy,
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_loss': self.best_loss,
            'policy_history': self.adaptive_optimizer.policy_history
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load training state"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.adaptive_optimizer.current_policy = checkpoint['adaptive_policy']
        self.step_count = checkpoint['step_count']
        self.epoch_count = checkpoint['epoch_count']
        self.best_loss = checkpoint['best_loss']
        self.adaptive_optimizer.policy_history = checkpoint['policy_history']