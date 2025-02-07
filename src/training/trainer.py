import torch
import wandb
import os
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from src.training.optim import HyperbolicOptimizer
from src.model.transformer import HyperbolicTransformer
from src.core.config.configurations import TrainingConfig

class Trainer:
    """Enhanced model trainer with monitoring"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = HyperbolicOptimizer(model, config)
        
        # Setup logging
        if config.use_wandb:
            wandb.init(project="hyperbolic-transformer")
            
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def train(self,
            train_dataloader: torch.utils.data.DataLoader,
            eval_dataloader: Optional[torch.utils.data.DataLoader] = None):
        """Train the model"""
        device = next(self.model.parameters()).device
        global_step = 0
        total_loss = 0
        logging_loss = 0
        
        try:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
                
                for step, batch in enumerate(epoch_iterator):
                    try:
                        # Move batch to device
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        
                        # Forward pass
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.config.accumulation_steps
                        
                        # Backward pass with gradient scaling
                        loss.backward()
                        
                        # Update running loss
                        total_loss += loss.item()
                        
                        # Update parameters
                        if (step + 1) % self.config.accumulation_steps == 0:
                            # Clip gradients
                            if self.config.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.max_grad_norm
                                )
                            
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            global_step += 1
                            
                            # Logging
                            if global_step % self.config.logging_steps == 0:
                                avg_loss = (total_loss - logging_loss) / self.config.logging_steps
                                logging_loss = total_loss
                                
                                metrics = {
                                    'loss': avg_loss,
                                    'learning_rate': self.optimizer.euclidean_optimizer.param_groups[0]['lr'],
                                    'epoch': epoch,
                                    'step': global_step,
                                    'memory_used': torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
                                }
                                
                                if self.config.use_wandb:
                                    wandb.log(metrics)
                                
                                epoch_iterator.set_postfix({'loss': f"{avg_loss:.4f}"})
                            
                            # Evaluation
                            if eval_dataloader is not None and global_step % self.config.evaluation_steps == 0:
                                eval_metrics = self.evaluate(eval_dataloader)
                                
                                if self.config.use_wandb:
                                    wandb.log(eval_metrics)
                                
                                # Resume training mode
                                self.model.train()
                            
                            # Save checkpoint
                            if global_step % self.config.save_steps == 0:
                                self.save_checkpoint(global_step)
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"\nWARNING: out of memory error. Skipping batch and clearing cache.")
                            if hasattr(self.optimizer, 'zero_grad'):
                                self.optimizer.zero_grad()
                            continue
                        raise e
                
                # End of epoch evaluation
                if eval_dataloader is not None:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint(global_step, "interrupted")
            raise
        
        except Exception as e:
            print(f"\nERROR during training: {str(e)}")
            self.save_checkpoint(global_step, "error")
            raise
        
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / global_step if global_step > 0 else float('inf')
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs['loss']
            total_loss += loss.item()
            total_steps += 1
            
        metrics = {
            'eval_loss': total_loss / total_steps,
            'perplexity': np.exp(total_loss / total_steps)
        }
        
        return metrics
    
    def save_checkpoint(self, step: int, prefix: str = "checkpoint") -> None:
        """Save model checkpoint
        
        Args:
            step (int): Current training step
            prefix (str, optional): Prefix for checkpoint filename. Defaults to "checkpoint".
        """
        if not self.config.checkpoint_dir:
            return
            
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"{prefix}_step_{step}.pt"
        )
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': {
                'euclidean': self.optimizer.euclidean_optimizer.state_dict(),
                'hyperbolic': self.optimizer.hyperbolic_optimizer.state_dict()
            },
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")