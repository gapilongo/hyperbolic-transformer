from typing import Dict, List, Optional, Any, Callable
import datetime
from collections import defaultdict
import plotly.graph_objects as go


class PipelineMonitor:
    """Monitor pipeline execution"""
    def __init__(self):
        self.history = defaultdict(list)
        self.stats = defaultdict(lambda: {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_duration': 0
        })
        
    def record_completion(self,
                         pipeline_name: str,
                         success: bool,
                         duration: Optional[float] = None,
                         error: Optional[str] = None):
        """Record pipeline completion"""
        completion = {
            'timestamp': datetime.now(),
            'success': success,
            'duration': duration,
            'error': error
        }
        
        self.history[pipeline_name].append(completion)
        
        # Update statistics
        stats = self.stats[pipeline_name]
        stats['total_runs'] += 1
        if success:
            stats['successful_runs'] += 1
        else:
            stats['failed_runs'] += 1
            
        if duration is not None:
            stats['average_duration'] = (
                (stats['average_duration'] * (stats['total_runs'] - 1) + duration) /
                stats['total_runs']
            )
            
    def get_pipeline_stats(self,
                          pipeline_name: str) -> Dict[str, Any]:
        """Get statistics for specific pipeline"""
        if pipeline_name not in self.stats:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
            
        return {
            'stats': self.stats[pipeline_name],
            'history': self.history[pipeline_name]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pipelines"""
        return {
            name: self.get_pipeline_stats(name)
            for name in self.stats
        }
    
    def plot_success_rates(self) -> go.Figure:
        """Plot pipeline success rates"""
        fig = go.Figure()
        
        for name, stats in self.stats.items():
            success_rate = (stats['successful_runs'] / stats['total_runs'] 
                          if stats['total_runs'] > 0 else 0)
            
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=[name],
                    y=[success_rate],
                    text=[f"{success_rate:.1%}"],
                    textposition='auto'
                )
            )
            
        fig.update_layout(
            title='Pipeline Success Rates',
            yaxis=dict(
                title='Success Rate',
                range=[0, 1],
                tickformat='%'
            ),
            showlegend=False
        )
        
        return fig