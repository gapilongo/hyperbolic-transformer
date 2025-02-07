import wandb
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import pandas as pd


class ExperimentAnalyzer:
    """Analyze experiment results"""
    def __init__(self,
                 project_name: str,
                 experiment_name: Optional[str] = None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # Load experiment data
        self.api = wandb.Api()
        self.runs = self._load_runs()
        
    def _load_runs(self) -> List[wandb.Run]:
        """Load experiment runs from W&B"""
        filters = {
            'project': self.project_name
        }
        
        if self.experiment_name:
            filters['name'] = self.experiment_name
            
        return list(self.api.runs(**filters))
    
    def compare_runs(self) -> pd.DataFrame:
        """Compare metrics across runs"""
        comparison = []
        
        for run in self.runs:
            run_data = {
                'id': run.id,
                'name': run.name,
                'status': run.state,
                'created': run.created_at,
                **run.summary._json_dict
            }
            comparison.append(run_data)
            
        return pd.DataFrame(comparison)
    
    def analyze_best_run(self,
                        metric: str,
                        higher_better: bool = False) -> Dict[str, Any]:
        """Analyze best performing run"""
        comparison = self.compare_runs()
        
        best_idx = (comparison[metric].argmax() if higher_better 
                   else comparison[metric].argmin())
        best_run = self.runs[best_idx]
        
        # Load full history
        history = pd.DataFrame(best_run.history())
        
        # Compute statistics
        metric_stats = {
            'mean': history[metric].mean(),
            'std': history[metric].std(),
            'min': history[metric].min(),
            'max': history[metric].max(),
            'final': history[metric].iloc[-1]
        }
        
        # Load artifacts
        artifacts = {}
        for artifact in best_run.logged_artifacts():
            artifacts[artifact.name] = artifact
            
        return {
            'run_id': best_run.id,
            'config': best_run.config,
            'metric_stats': metric_stats,
            'history': history,
            'artifacts': artifacts
        }
    
    def plot_metric_comparison(self,
                             metric: str,
                             smoothing: float = 0.0) -> go.Figure:
        """Plot metric comparison across runs"""
        fig = go.Figure()
        
        for run in self.runs:
            history = pd.DataFrame(run.history())
            values = history[metric]
            
            if smoothing > 0:
                values = values.ewm(alpha=1-smoothing).mean()
                
            fig.add_trace(
                go.Scatter(
                    y=values,
                    name=run.name,
                    mode='lines'
                )
            )
            
        fig.update_layout(
            title=f'{metric} Comparison',
            xaxis_title='Step',
            yaxis_title=metric,
            hovermode='x unified'
        )
        
        return fig