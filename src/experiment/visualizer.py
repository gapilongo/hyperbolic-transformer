from typing import Dict, List, Optional, Any
import plotly.graph_objects as go


class ExperimentVisualizer:
    """Create experiment visualizations"""
    def create_visualizations(self,
                            metrics_history: Dict[str, List[float]]) -> Dict[str, go.Figure]:
        """Create standard visualizations"""
        figures = {}
        
        # Training curves
        figures['training_curves'] = self._plot_training_curves(metrics_history)
        
        # Learning rate schedule
        if 'learning_rate' in metrics_history:
            figures['lr_schedule'] = self._plot_lr_schedule(
                metrics_history['learning_rate']
            )
            
        # Loss landscape
        if all(k in metrics_history for k in ['loss', 'grad_norm']):
            figures['loss_landscape'] = self._plot_loss_landscape(
                metrics_history['loss'],
                metrics_history['grad_norm']
            )
            
        return figures
    
    def _plot_training_curves(self,
                            metrics_history: Dict[str, List[float]]) -> go.Figure:
        """Plot training metrics over time"""
        fig = go.Figure()
        
        for metric, values in metrics_history.items():
            if metric not in ['learning_rate', 'grad_norm']:  # Plot these separately
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        name=metric,
                        mode='lines'
                    )
                )
                
        fig.update_layout(
            title='Training Metrics',
            xaxis_title='Step',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def _plot_lr_schedule(self, lr_history: List[float]) -> go.Figure:
        """Plot learning rate schedule"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                y=lr_history,
                name='learning_rate',
                mode='lines'
            )
        )
        
        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Step',
            yaxis_title='Learning Rate',
            yaxis_type='log'
        )
        
        return fig
    
    def _plot_loss_landscape(self,
                           loss_history: List[float],
                           grad_history: List[float]) -> go.Figure:
        """Plot loss landscape visualization"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=grad_history,
                y=loss_history,
                mode='markers',
                marker=dict(
                    size=5,
                    color=list(range(len(loss_history))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Step')
                )
            )
        )
        
        fig.update_layout(
            title='Loss Landscape',
            xaxis_title='Gradient Norm',
            yaxis_title='Loss',
            hovermode='closest'
        )
        
        return fig