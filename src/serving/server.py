import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from core.config.configurations import ServingConfig
from optimizer import ServingOptimizer
from data.tokenizer import EnhancedTokenizer


class InferenceServer:
    """Serve model for inference"""
    def __init__(self,
                 model_path: str,
                 config: ServingConfig):
        self.config = config
        
        # Load components
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer(model_path)
        self.optimizer = ServingOptimizer(self.model, config)
        
        # Initialize workers
        self.workers = []
        self._start_workers()
        
    def _load_model(self, path: str) -> torch.nn.Module:
        """Load model and optimize for inference"""
        model = torch.jit.load(path)
        model.eval()
        
        if torch.cuda.is_available():
            model.cuda()
            
        return model
    
    def _load_tokenizer(self, model_path: str) -> EnhancedTokenizer:
        """Load tokenizer"""
        tokenizer_path = Path(model_path).parent / 'tokenizer'
        return EnhancedTokenizer.load(str(tokenizer_path))
    
    def _start_workers(self):
        """Start worker processes"""
        for _ in range(self.config.num_workers):
            worker = InferenceWorker(
                self.model,
                self.tokenizer,
                self.optimizer,
                self.config
            )
            self.workers.append(worker)
            
    def serve(self, text: str) -> str:
        """Serve inference request"""
        # Preprocess input
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Get available worker
        worker = self._get_available_worker()
        
        # Process request
        outputs = worker.process_request(inputs)
        
        # Decode outputs
        return self.tokenizer.decode(
            outputs['logits'].argmax(dim=-1).squeeze()
        )
        
    def _get_available_worker(self) -> 'InferenceWorker':
        """Get available worker using round-robin"""
        worker = self.workers.pop(0)
        self.workers.append(worker)
        return worker
    
    def stop(self):
        """Stop all workers"""
        for worker in self.workers:
            worker.stop()
            
class InferenceWorker:
    """Worker process for handling inference requests"""
    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer: EnhancedTokenizer,
                 optimizer: ServingOptimizer,
                 config: ServingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        
        self.is_running = True
        
    def process_request(self,
                       inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process single inference request"""
        # Apply optimizations
        optimized_outputs = self.optimizer.optimize_inference(inputs)
        
        return optimized_outputs
    
    def stop(self):
        """Stop worker"""
        self.is_running = False