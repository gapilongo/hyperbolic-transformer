from dataclasses import dataclass
import logging
from collections import defaultdict
import GPUtil
import torch
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
from torch.utils.tensorboard import SummaryWriter
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer
@dataclass
class PerformanceMetrics:
    """Track various performance metrics"""
    batch_time: float
    memory_used: float
    gpu_utilization: float
    throughput: float
    gpu_memory: float
    cpu_utilization: float

class PerformanceMonitor:
    """Monitor and optimize model performance"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig,
                 log_dir: str = "logs"):
        self.model = model
        self.config = config
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir)
        
        # Performance tracking
        self.metrics_history = defaultdict(list)
        self.current_batch_size = config.batch_size
        self.min_batch_size = 4
        self.max_batch_size = 128
        
        # Resource monitoring
        self.start_time = time.time()
        self.step_count = 0
        
    def step(self, loss: float, batch_size: int):
        """Record performance metrics for one step"""
        metrics = self._collect_metrics(loss, batch_size)
        self._update_history(metrics)
        self._log_metrics(metrics)
        
        # Adjust batch size if needed
        self._adjust_batch_size(metrics)
        
        self.step_count += 1
        
    def _collect_metrics(self,
                        loss: float,
                        batch_size: int) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # Get GPU metrics if available
        gpu_metrics = GPUtil.getGPUs()[0] if torch.cuda.is_available() else None
        
        metrics = PerformanceMetrics(
            batch_time=time.time() - self.start_time,
            memory_used=psutil.Process().memory_info().rss / 1024 ** 2,  # MB
            gpu_utilization=gpu_metrics.load if gpu_metrics else 0,
            throughput=batch_size / (time.time() - self.start_time),
            gpu_memory=gpu_metrics.memoryUsed if gpu_metrics else 0,
            cpu_utilization=psutil.cpu_percent()
        )
        
        self.start_time = time.time()
        return metrics
    
    def _update_history(self, metrics: PerformanceMetrics):
        """Update metrics history"""
        for key, value in metrics.__dict__.items():
            self.metrics_history[key].append(value)
            
            # Keep history manageable
            if len(self.metrics_history[key]) > 1000:
                self.metrics_history[key].pop(0)
                
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log metrics to tensorboard and console"""
        # Log to tensorboard
        for key, value in metrics.__dict__.items():
            self.writer.add_scalar(f"performance/{key}", value, self.step_count)
            
        # Log to console periodically
        if self.step_count % 100 == 0:
            self.logger.info(
                f"Step {self.step_count}: "
                f"Throughput={metrics.throughput:.2f} samples/sec, "
                f"GPU Util={metrics.gpu_utilization:.1f}%, "
                f"Memory={metrics.memory_used:.1f}MB"
            )
            
    def _adjust_batch_size(self, metrics: PerformanceMetrics):
        """Dynamically adjust batch size based on performance"""
        if self.step_count < 100:  # Wait for warm-up
            return
            
        # Check memory pressure
        memory_pressure = metrics.gpu_memory / self._get_total_gpu_memory()
        
        if memory_pressure > 0.9:  # High memory pressure
            self._decrease_batch_size()
        elif memory_pressure < 0.7 and metrics.gpu_utilization > 80:
            self._increase_batch_size()
            
    def _decrease_batch_size(self):
        """Decrease batch size"""
        new_size = max(self.min_batch_size, self.current_batch_size // 2)
        if new_size != self.current_batch_size:
            self.logger.info(f"Decreasing batch size to {new_size}")
            self.current_batch_size = new_size
            
    def _increase_batch_size(self):
        """Increase batch size"""
        new_size = min(self.max_batch_size, self.current_batch_size * 2)
        if new_size != self.current_batch_size:
            self.logger.info(f"Increasing batch size to {new_size}")
            self.current_batch_size = new_size
            
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in MB"""
        if torch.cuda.is_available():
            return GPUtil.getGPUs()[0].memoryTotal
        return 0
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics"""
        summary = {}
        
        for key, values in self.metrics_history.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)
            
        return summary