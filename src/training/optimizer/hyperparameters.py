import optuna
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from core.config.configurations import OptimizationConfig

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    def __init__(self,
                 model_class: type,
                 config: OptimizationConfig,
                 train_fn: Callable,
                 eval_fn: Callable):
        self.model_class = model_class
        self.config = config
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        
        # Initialize study
        self.study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=config.pruner_warmup_steps
            ),
            storage=config.storage_url,
            load_if_exists=True
        )
        
        # Track best configuration
        self.best_params = None
        self.best_score = float('inf')
        
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        self.study.optimize(
            func=self._objective,
            n_trials=self.config.n_trials,
            callbacks=[self._optimization_callback]
        )
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.study.trials_dataframe()
        }
        
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        params = self._sample_parameters(trial)
        
        # Create model with sampled parameters
        model = self.model_class(**params)
        
        # Train and evaluate
        for epoch in range(self.config.epochs_per_trial):
            train_metrics = self.train_fn(model, params)
            eval_metrics = self.eval_fn(model)
            
            # Report intermediate objective value
            trial.report(eval_metrics[self.config.optimization_metric], epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return eval_metrics[self.config.optimization_metric]
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for trial"""
        params = {
            'learning_rate': trial.suggest_loguniform(
                'learning_rate',
                *self.config.lr_range
            ),
            'batch_size': trial.suggest_int(
                'batch_size',
                *self.config.batch_size_range,
                log=True
            ),
            'weight_decay': trial.suggest_loguniform(
                'weight_decay',
                *self.config.weight_decay_range
            ),
            'dropout': trial.suggest_uniform(
                'dropout',
                *self.config.dropout_range
            ),
            # Model architecture parameters
            'num_layers': trial.suggest_int('num_layers', 4, 12),
            'hidden_size': trial.suggest_categorical(
                'hidden_size',
                [256, 512, 768, 1024]
            ),
            'num_attention_heads': trial.suggest_categorical(
                'num_attention_heads',
                [4, 8, 12, 16]
            ),
            # Hyperbolic space parameters
            'hyperbolic_dim': trial.suggest_int('hyperbolic_dim', 2, 8),
            'curvature': trial.suggest_uniform('curvature', -2.0, -0.1)
        }
        
        return params
    
    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback after each trial"""
        if trial.value < self.best_score:
            self.best_score = trial.value
            self.best_params = trial.params