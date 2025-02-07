import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
import os
import multiprocessing as mp


class WorkerManager:
    """Manage worker processes and inter-process communication"""
    def __init__(self,
                 world_size: int,
                 backend: str = 'nccl'):
        self.world_size = world_size
        self.backend = backend
        
        # Process tracking
        self.processes = []
        self.active_workers = set()
        
        # Communication queues
        self.command_queues = {}
        self.result_queues = {}
        
    def start_workers(self):
        """Start worker processes"""
        for rank in range(self.world_size):
            queue_pair = self._create_queue_pair()
            process = mp.Process(
                target=self._worker_process,
                args=(rank, *queue_pair)
            )
            process.start()
            self.processes.append(process)
            self.active_workers.add(rank)
            
    def _create_queue_pair(self) -> Tuple[mp.Queue, mp.Queue]:
        """Create command and result queues for a worker"""
        return mp.Queue(), mp.Queue()
    
    def _worker_process(self,
                       rank: int,
                       command_queue: mp.Queue,
                       result_queue: mp.Queue):
        """Worker process main loop"""
        try:
            # Initialize distributed environment
            self._init_worker(rank)
            
            while True:
                command = command_queue.get()
                if command == "stop":
                    break
                    
                # Process command
                result = self._process_command(command)
                result_queue.put(result)
                
        except Exception as e:
            result_queue.put(("error", str(e)))
            
        finally:
            # Cleanup
            if dist.is_initialized():
                dist.destroy_process_group()
                
    def _init_worker(self, rank: int):
        """Initialize worker's distributed environment"""
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        dist.init_process_group(
            backend=self.backend,
            init_method='env://'
        )
        
    def _process_command(self, command: Dict[str, Any]) -> Any:
        """Process a command received from main process"""
        command_type = command['type']
        
        if command_type == 'forward':
            return self._forward_pass(command['data'])
        elif command_type == 'backward':
            return self._backward_pass(command['gradients'])
        elif command_type == 'sync':
            return self._sync_parameters()
        else:
            raise ValueError(f"Unknown command type: {command_type}")
            
    def send_command(self,
                    rank: int,
                    command: Dict[str, Any]) -> Any:
        """Send command to specific worker"""
        if rank not in self.active_workers:
            raise ValueError(f"Worker {rank} is not active")
            
        self.command_queues[rank].put(command)
        return self.result_queues[rank].get()
    
    def broadcast_command(self,
                         command: Dict[str, Any]) -> List[Any]:
        """Send command to all workers"""
        results = []
        for rank in self.active_workers:
            results.append(self.send_command(rank, command))
        return results
    
    def stop_workers(self):
        """Stop all worker processes"""
        # Send stop command to all workers
        for rank in self.active_workers:
            self.command_queues[rank].put("stop")
            
        # Wait for processes to finish
        for process in self.processes:
            process.join()
            
        self.active_workers.clear()
        self.processes.clear()