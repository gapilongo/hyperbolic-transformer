import torch
import onnx
import tensorrt as trt
from typing import Dict, List, Optional, Union, Any
from torch.jit import trace
from core.config.configurations import ServingConfig
import time
from utils.memory import LRUCache


class ServingOptimizer:
    """Optimize model for inference"""
    def __init__(self, 
                 model: Union[str, torch.nn.Module],
                 config: ServingConfig):
        self.config = config
        
        # Load model if path provided
        self.model = (
            self._load_model(model)
            if isinstance(model, str)
            else model
        )
        
        # Initialize optimizations
        self.request_queue = []
        self.response_cache = LRUCache(config.cache_size)
        self.current_batch = []
        self.last_batch_time = time.time()
        
    def _load_model(self, path: str) -> torch.nn.Module:
        """Load model from file"""
        if path.endswith('.pt'):
            return torch.jit.load(path)
        elif path.endswith('.onnx'):
            return onnx.load(path)
        elif path.endswith('.trt'):
            return self._load_tensorrt(path)
        else:
            raise ValueError(f"Unsupported model format: {path}")
            
    def _load_tensorrt(self, path: str) -> trt.ICudaEngine:
        """Load TensorRT engine"""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(path, 'rb') as f:
            engine_bytes = f.read()
            
        return runtime.deserialize_cuda_engine(engine_bytes)
    
    def optimize_inference(self,
                         request: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply inference optimizations"""
        # Check cache
        cache_key = self._get_cache_key(request)
        cached_response = self.response_cache.get(cache_key)
        if cached_response is not None:
            return cached_response
            
        # Add to current batch
        self.current_batch.append(request)
        
        # Check if batch should be processed
        if self._should_process_batch():
            return self._process_batch()
            
        # Wait for more requests
        return self._wait_for_batch()
    
    def _get_cache_key(self, request: Dict[str, torch.Tensor]) -> str:
        """Generate cache key for request"""
        return str(hash(tuple(
            tensor.numpy().tobytes() 
            for tensor in request.values()
        )))
        
    def _should_process_batch(self) -> bool:
        """Check if current batch should be processed"""
        if len(self.current_batch) >= self.config.batch_size:
            return True
            
        if time.time() - self.last_batch_time > self.config.max_batch_delay:
            return True
            
        return False
    
    def _process_batch(self) -> Dict[str, torch.Tensor]:
        """Process current batch"""
        # Combine inputs
        batch_inputs = self._combine_inputs(self.current_batch)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            
        # Split outputs
        individual_outputs = self._split_outputs(outputs, len(self.current_batch))
        
        # Cache results
        for request, output in zip(self.current_batch, individual_outputs):
            cache_key = self._get_cache_key(request)
            self.response_cache.put(cache_key, output)
            
        # Reset batch state
        self.current_batch = []
        self.last_batch_time = time.time()
        
        return individual_outputs
    
    def _wait_for_batch(self) -> Dict[str, torch.Tensor]:
        """Wait for batch to be ready"""
        start_time = time.time()
        
        while not self._should_process_batch():
            if time.time() - start_time > self.config.timeout:
                # Timeout reached, process current batch
                return self._process_batch()
            time.sleep(0.001)  # Small delay
            
        return self._process_batch()
    
    def _combine_inputs(self,
                       requests: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine individual requests into batch"""
        combined = {}
        for key in requests[0].keys():
            combined[key] = torch.cat([r[key] for r in requests])
        return combined
    
    def _split_outputs(self,
                      outputs: Dict[str, torch.Tensor],
                      batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split batch outputs into individual responses"""
        individual = []
        for i in range(batch_size):
            individual.append({
                key: tensor[i:i+1]
                for key, tensor in outputs.items()
            })
        return individual