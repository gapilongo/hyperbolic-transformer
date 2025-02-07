import unittest
import torch
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from src.core.config.configurations import ModelConfig, TestConfig
from src.model.transformer import HyperbolicTransformer

class PerformanceTests(unittest.TestCase):
    """Performance and stress testing"""
    
    def setUp(self):
        """Setup performance tests"""
        self.config = TestConfig(
            test_data_path="tests/data",
            model_path="tests/models"
        )
        self.model = HyperbolicTransformer(ModelConfig())
        
    def test_inference_latency(self):
        """Test inference latency"""
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                # Create dummy batch
                batch = {
                    'input_ids': torch.randint(
                        0, 1000,
                        (batch_size, seq_length)
                    ),
                    'attention_mask': torch.ones(batch_size, seq_length)
                }
                
                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(**batch)
                duration = time.time() - start_time
                
                # Check against threshold
                self.assertLess(
                    duration,
                    self.config.performance_threshold,
                    f"Inference too slow for batch_size={batch_size}, "
                    f"seq_length={seq_length}"
                )
                
    def test_memory_usage(self):
        """Test memory usage under load"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        initial_memory = torch.cuda.memory_allocated()
        
        # Run model with increasing batch sizes
        for batch_size in self.config.batch_sizes:
            batch = {
                'input_ids': torch.randint(
                    0, 1000,
                    (batch_size, 512)
                ).cuda(),
                'attention_mask': torch.ones(batch_size, 512).cuda()
            }
            
            with torch.no_grad():
                _ = self.model(**batch)
                
            # Check memory usage
            current_memory = torch.cuda.memory_allocated()
            memory_used = current_memory / torch.cuda.get_device_properties(0).total_memory
            
            self.assertLess(
                memory_used,
                self.config.gpu_memory_threshold,
                f"Memory usage too high for batch_size={batch_size}"
            )
            
    def test_stress_concurrent_requests(self):
        """Test model under concurrent load"""
        def inference_worker():
            batch = {
                'input_ids': torch.randint(0, 1000, (1, 128)),
                'attention_mask': torch.ones(1, 128)
            }
            
            with torch.no_grad():
                _ = self.model(**batch)
                
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [
                executor.submit(inference_worker)
                for _ in range(self.config.num_stress_iterations)
            ]
            
            # Check all requests completed successfully
            for future in futures:
                self.assertIsNone(future.exception())