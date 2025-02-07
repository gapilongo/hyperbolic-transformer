import torch
from core.config.configurations import ModelConfig

class GradientOptimizer:
    """Optimize gradient computation and synchronization"""
    def __init__(self,
                 model: torch.nn.Module,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
        # Gradient accumulation
        self.grad_accum_steps = 1
        self.current_step = 0
        
        # Gradient clipping
        self.clip_value = config.gradient_clip
        self.clip_norm = True
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = True
        
    def backward_pass(self, loss: torch.Tensor):
        """Optimized backward pass with gradient accumulation"""
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.grad_accum_steps
        
        if self.use_amp:
            scaled_loss = self.scaler.scale(scaled_loss)
            
        # Backward pass
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Return if still accumulating
        if self.current_step < self.grad_accum_steps:
            return
        
        # Unscale gradients for clipping with AMP
        if self.use_amp:
            self.scaler.unscale_(self.model.optimizer)
            
        # Clip gradients
        if self.clip_value > 0:
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_value
                )
            else:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.clip_value
                )
                
        # Optimizer step with AMP
        if self.use_amp:
            self.scaler.step(self.model.optimizer)
            self.scaler.update()
        else:
            self.model.optimizer.step()
            
        # Reset gradients and step counter
        self.model.optimizer.zero_grad()
        self.current_step = 0
        
    def get_grad_norm(self) -> float:
        """Compute gradient norm"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5