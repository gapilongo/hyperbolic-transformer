import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from collections import defaultdict
import time
from typing import Dict, List, Optional, Any, Tuple
import os
from core.config.configurations import ModelConfig, DistributedConfig
from model.transformer import HyperbolicTransformer


class DistributedTrainer:
    """Manage distributed training across multiple devices"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: ModelConfig,
                 dist_config: DistributedConfig):
        self.model = model
        self.config = config
        self.dist_config = dist_config
        
        # Initialize distributed environment
        self.init_distributed()
        
        # Wrap model for distributed training
        self.model = self.prepare_model()
        
        # Performance tracking
        self.sync_metrics = defaultdict(list)
        self.communication_stats = defaultdict(list)
        
    def init_distributed(self):
        """Initialize distributed environment"""
        if self.dist_config.use_horovod:
            self._init_horovod()
        else:
            self._init_pytorch_distributed()
            
    def _init_pytorch_distributed(self):
        """Initialize PyTorch distributed backend"""
        os.environ['MASTER_ADDR'] = self.dist_config.master_addr
        os.environ['MASTER_PORT'] = self.dist_config.master_port
        
        dist.init_process_group(
            backend=self.dist_config.backend,
            world_size=self.dist_config.world_size,
            rank=self.dist_config.rank
        )
        
        # Set device
        torch.cuda.set_device(self.dist_config.local_rank)
        
    def _init_horovod(self):
        """Initialize Horovod backend"""
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        
    def prepare_model(self) -> torch.nn.Module:
        """Prepare model for distributed training"""
        self.model = self.model.cuda()
        
        if self.dist_config.use_horovod:
            self.model = hvd.DistributedOptimizer(
                self.model,
                named_parameters=self.model.named_parameters()
            )
        else:
            self.model = DDP(
                self.model,
                device_ids=[self.dist_config.local_rank],
                output_device=self.dist_config.local_rank,
                find_unused_parameters=True
            )
            
        return self.model
    
    def prepare_dataloader(self, 
                          dataset: torch.utils.data.Dataset,
                          batch_size: int) -> torch.utils.data.DataLoader:
        """Prepare dataloader for distributed training"""
        if self.dist_config.use_horovod:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=hvd.size(),
                rank=hvd.rank()
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist_config.world_size,
                rank=self.dist_config.rank
            )
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def sync_gradients(self):
        """Synchronize gradients across devices"""
        if self.dist_config.use_horovod:
            self._sync_horovod_gradients()
        else:
            self._sync_pytorch_gradients()
            
    def _sync_horovod_gradients(self):
        """Synchronize gradients using Horovod"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                start_time = time.time()
                tensor = param.grad.data
                
                # Perform allreduce
                avg_tensor = hvd.allreduce(tensor, name=name)
                param.grad.data = avg_tensor
                
                # Track communication time
                self.communication_stats[name].append(time.time() - start_time)
                
    def _sync_pytorch_gradients(self):
        """Synchronize gradients using PyTorch DDP"""
        start_time = time.time()
        torch.distributed.barrier()
        self.communication_stats['barrier_time'].append(time.time() - start_time)
        
    def broadcast_state_dict(self):
        """Broadcast model state from rank 0 to all other ranks"""
        if self.dist_config.use_horovod:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        else:
            for name, param in self.model.state_dict().items():
                dist.broadcast(param, src=0)
                
    def reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reduce metrics across all devices"""
        reduced_metrics = {}
        
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                start_time = time.time()
                
                if self.dist_config.use_horovod:
                    reduced_value = hvd.allreduce(value, name=f"metric_{name}")
                else:
                    reduced_value = value.clone()
                    dist.all_reduce(reduced_value)
                    reduced_value /= self.dist_config.world_size
                    
                reduced_metrics[name] = reduced_value
                self.sync_metrics[name].append(time.time() - start_time)
                
        return reduced_metrics
    
    def cleanup(self):
        """Clean up distributed environment"""
        if not self.dist_config.use_horovod:
            dist.destroy_process_group()