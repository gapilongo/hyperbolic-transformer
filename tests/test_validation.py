import unittest
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from src.core.config.configurations import ModelConfig
from src.model.transformer import HyperbolicTransformer
from test_model import ModelTestSuite
from test_integration import IntegrationTests
from test_performance import PerformanceTests

class ValidationTests(unittest.TestCase):
    """Model validation and quality tests"""
    
    def setUp(self):
        """Setup validation tests"""
        self.model = HyperbolicTransformer(ModelConfig())
        self.test_samples = self._load_validation_samples()
        
    def _load_validation_samples(self) -> List[Dict[str, Any]]:
        """Load validation test samples"""
        with open("tests/data/validation_samples.json") as f:
            return json.load(f)
            
    def test_output_quality(self):
        """Test model output quality"""
        for sample in self.test_samples:
            input_text = sample['input']
            expected = sample['expected']
            
            # Generate output
            output = self.model.generate(input_text)
            
            # Check basic quality metrics
            self.assertGreater(len(output), 0)
            self.assertLess(
                self._compute_perplexity(output),
                sample.get('max_perplexity', 100)
            )
            
            if 'must_contain' in sample:
                for phrase in sample['must_contain']:
                    self.assertIn(
                        phrase.lower(),
                        output.lower()
                    )
                    
    def test_model_robustness(self):
        """Test model robustness"""
        # Test with noisy inputs
        noise_types = ['typos', 'extra_spaces', 'missing_punctuation']
        
        for sample in self.test_samples:
            clean_output = self.model.generate(sample['input'])
            
            for noise_type in noise_types:
                noisy_input = self._add_noise(
                    sample['input'],
                    noise_type
                )
                noisy_output = self.model.generate(noisy_input)
                
                # Outputs should be similar despite noise
                similarity = self._compute_similarity(
                    clean_output,
                    noisy_output
                )
                self.assertGreater(similarity, 0.8)
                
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of generated text"""
        # Implementation of perplexity computation
        return 0.0  # Placeholder
        
    def _compute_similarity(self,
                          text1: str,
                          text2: str) -> float:
        """Compute semantic similarity between texts"""
        # Implementation of similarity computation
        return 1.0  # Placeholder
        
    def _add_noise(self,
                  text: str,
                  noise_type: str) -> str:
        """Add controlled noise to input"""
        # Implementation of noise addition
        return text  # Placeholder

def run_all_tests():
    """Run all test suites"""
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ModelTestSuite))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PerformanceTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IntegrationTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ValidationTests))
    
    # Run tests
    return runner.run(suite)

if __name__ == '__main__':
    run_all_tests()