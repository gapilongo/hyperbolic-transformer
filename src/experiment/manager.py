import wandb
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import torch
from visualizer import ExperimentVisualizer
from core.config.configurations import ExperimentConfig

class ExperimentManager:
    """Manage and track experiments"""
    def __init__(self,
                 config: ExperimentConfig):
        self.config = config
        
        # Initialize tracking
        self.run = wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            tags=config.tags,
            config=config.__dict__
        )
        
        # Track experiment state
        self.step_count = 0
        self.metrics_history = defaultdict(list)
        self.artifacts = {}
        self.checkpoints = []
        
        # Results visualization
        self.visualizer = ExperimentVisualizer()
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        self.step_count += 1
        
        # Update history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
        # Log to wandb at specified intervals
        if self.step_count % self.config.log_interval == 0:
            wandb.log(metrics, step=self.step_count)
            
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       extra_state: Optional[Dict] = None):
        """Save model checkpoint"""
        if not self.config.save_checkpoints:
            return
            
        if self.step_count % self.config.checkpoint_interval != 0:
            return
            
        # Create checkpoint
        checkpoint = {
            'step': self.step_count,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': dict(self.metrics_history),
            'extra_state': extra_state or {}
        }
        
        # Save locally
        checkpoint_path = Path(f"checkpoints/checkpoint_{self.step_count}.pt")
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Log to wandb
        if self.config.track_artifacts:
            artifact = wandb.Artifact(
                name=f"checkpoint_{self.step_count}",
                type='model'
            )
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)
            
        self.checkpoints.append(checkpoint_path)
        
    def log_artifact(self, name: str, artifact: Any):
        """Log arbitrary artifact"""
        if not self.config.track_artifacts:
            return
            
        # Save artifact
        artifact_path = Path(f"artifacts/{name}")
        artifact_path.parent.mkdir(exist_ok=True)
        
        if isinstance(artifact, (dict, list)):
            with open(artifact_path.with_suffix('.json'), 'w') as f:
                json.dump(artifact, f, indent=2)
        elif isinstance(artifact, pd.DataFrame):
            artifact.to_csv(artifact_path.with_suffix('.csv'))
        else:
            with open(artifact_path.with_suffix('.txt'), 'w') as f:
                f.write(str(artifact))
                
        # Log to wandb
        wandb_artifact = wandb.Artifact(
            name=name,
            type='artifact'
        )
        wandb_artifact.add_file(str(artifact_path))
        self.run.log_artifact(wandb_artifact)
        
        self.artifacts[name] = artifact_path
        
    def visualize_results(self):
        """Generate experiment visualizations"""
        if not self.config.visualize_results:
            return
            
        # Create visualizations
        figures = self.visualizer.create_visualizations(self.metrics_history)
        
        # Save locally
        viz_path = Path("visualizations")
        viz_path.mkdir(exist_ok=True)
        
        for name, fig in figures.items():
            fig_path = viz_path / f"{name}.html"
            fig.write_html(str(fig_path))
            
            # Log to wandb
            wandb.log({name: fig})
            
    def finish(self):
        """Finish experiment tracking"""
        # Generate final visualizations
        self.visualize_results()
        
        # Save final metrics
        self.log_artifact(
            "final_metrics",
            dict(self.metrics_history)
        )
        
        # Log experiment summary
        summary = {
            'total_steps': self.step_count,
            'final_metrics': {
                key: values[-1]
                for key, values in self.metrics_history.items()
            },
            'num_checkpoints': len(self.checkpoints),
            'num_artifacts': len(self.artifacts)
        }
        
        for key, value in summary.items():
            wandb.run.summary[key] = value
            
        wandb.finish()



