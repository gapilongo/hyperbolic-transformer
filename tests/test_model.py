import unittest
from typing import Dict, List, Optional, Any
import torch
from pathlib import Path
import json
from src.core.config.configurations import ModelConfig, TokenizerConfig, TestConfig
from src.model.transformer import HyperbolicTransformer
from src.data.tokenizer import EnhancedTokenizer



class ModelTestSuite(unittest.TestCase):
    """Comprehensive test suite for model components"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.config = TestConfig(
            test_data_path="tests/data",
            model_path="tests/models"
        )
        
        # Initialize model and components
        cls.model = HyperbolicTransformer(ModelConfig())
        cls.tokenizer = EnhancedTokenizer(TokenizerConfig())
        
        # Load test data
        cls.test_data = cls._load_test_data()
        
    @classmethod
    def _load_test_data(cls) -> Dict[str, Any]:
        """Load test datasets"""
        test_data = {}
        data_path = Path(cls.config.test_data_path)
        
        # Load different test sets
        for data_file in data_path.glob("*.json"):
            with open(data_file) as f:
                test_data[data_file.stem] = json.load(f)
                
        return test_data
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, HyperbolicTransformer)
        self.assertEqual(self.model.config.hidden_size, 768)
        self.assertEqual(self.model.config.num_attention_heads, 12)
        
    def test_hyperbolic_operations(self):
        """Test hyperbolic space operations"""
        # Test distance computation
        x = torch.randn(3)
        y = torch.randn(3)
        distance = self.model.hyperbolic.distance(x, y)
        
        self.assertIsInstance(distance, torch.Tensor)
        self.assertTrue(distance >= 0)
        
        # Test exponential map
        result = self.model.hyperbolic.exp_map(x, y)
        self.assertEqual(result.shape, x.shape)
        
    def test_attention_mechanism(self):
        """Test attention computation"""
        batch_size = 4
        seq_length = 16
        hidden_size = 768
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Compute attention
        attention_output = self.model.attention(
            hidden_states,
            attention_mask
        )
        
        self.assertEqual(
            attention_output.shape,
            (batch_size, seq_length, hidden_size)
        )
        
    def test_tokenization(self):
        """Test tokenization pipeline"""
        text = "This is a test sentence."
        
        # Test encoding
        encoded = self.tokenizer.encode(text)
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)
        
        # Test decoding
        decoded = self.tokenizer.decode(encoded['input_ids'])
        self.assertIsInstance(decoded, str)
        
    def test_model_forward_pass(self):
        """Test model forward pass"""
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 16)),
            'attention_mask': torch.ones(4, 16)
        }
        
        outputs = self.model(**batch)
        
        self.assertIn('logits', outputs)
        self.assertIn('last_hidden_state', outputs)
        
    def test_loss_computation(self):
        """Test loss computation"""
        logits = torch.randn(4, 16, 1000)
        labels = torch.randint(0, 1000, (4, 16))
        
        loss = self.model.compute_loss(logits, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        