from collections import defaultdict
import GPUtil
import torch
import psutil
from typing import Dict, List, Optional, Any
import gc
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer

class MemoryManager:
    """Manage memory usage during training"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig):
        self.model = model
        self.config = config
        
        # Memory thresholds
        self.gpu_memory_threshold = 0.9  # 90% GPU memory utilization
        self.cpu_memory_threshold = 0.85  # 85% CPU memory utilization
        
        # Tracking
        self.memory_stats = defaultdict(list)
        
    def check_memory(self) -> bool:
        """Check memory usage and take action if needed"""
        stats = self._get_memory_stats()
        self._update_stats(stats)
        
        if self._should_optimize_memory(stats):
            self._optimize_memory()
            return True
            
        return False
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent() / 100,
            'cpu_memory': psutil.Process().memory_info().rss / psutil.virtual_memory().total
        }
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            stats.update({
                'gpu_memory': gpu.memoryUsed / gpu.memoryTotal,
                'gpu_utilization': gpu.load
            })
            
        return stats
    
    def _update_stats(self, stats: Dict[str, float]):
        """Update memory statistics"""
        for key, value in stats.items():
            self.memory_stats[key].append(value)
            
    def _should_optimize_memory(self, stats: Dict[str, float]) -> bool:
        """Check if memory optimization is needed"""
        return (
            stats.get('gpu_memory', 0) > self.gpu_memory_threshold or
            stats['cpu_memory'] > self.cpu_memory_threshold
        )
        
    def _optimize_memory(self):
        """Perform memory optimization"""
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear unused memory
        gc.collect()
        
        # Log optimization
        self.logger.info("Performed memory optimization")