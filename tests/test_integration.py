import unittest
import torch
from src.core.config.configurations import ModelConfig, TestConfig, TokenizerConfig, TrainingConfig, ServingConfig
from src.model.transformer import HyperbolicTransformer
from src.data.tokenizer import EnhancedTokenizer
from src.training.trainer import Trainer
from src.serving.server import InferenceServer
from src.data.dataset import TextDataset, create_dataloader



class IntegrationTests(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Setup integration tests"""
        self.config = TestConfig(
            test_data_path="tests/data",
            model_path="tests/models"
        )
        
        # Initialize all components
        self.model = HyperbolicTransformer(ModelConfig())
        self.tokenizer = EnhancedTokenizer(TokenizerConfig())
        self.trainer = Trainer(TrainingConfig())
        self.server = InferenceServer(ServingConfig())
        
    def test_training_pipeline(self):
        """Test complete training pipeline"""
        # Load sample dataset
        dataset = TextDataset(
            texts=["Sample text 1", "Sample text 2"],
            tokenizer=self.tokenizer
        )
        dataloader = create_dataloader(dataset, batch_size=2)
        
        # Train for one epoch
        metrics = self.trainer.train_epoch(
            model=self.model,
            dataloader=dataloader
        )
        
        self.assertIn('loss', metrics)
        self.assertTrue(metrics['loss'] > 0)
        
    def test_inference_pipeline(self):
        """Test complete inference pipeline"""
        text = "This is a test input."
        
        # Process through entire pipeline
        encoded = self.tokenizer.encode(text)
        outputs = self.model(**encoded)
        decoded = self.tokenizer.decode(
            outputs['logits'].argmax(dim=-1)
        )
        
        self.assertIsInstance(decoded, str)
        self.assertTrue(len(decoded) > 0)
        
    def test_model_export(self):
        """Test model export and loading"""
        # Export model
        export_path = "test_model.pt"
        torch.save(self.model.state_dict(), export_path)
        
        # Load exported model
        loaded_model = HyperbolicTransformer(ModelConfig())
        loaded_model.load_state_dict(torch.load(export_path))
        
        # Verify behavior matches
        test_input = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            original_output = self.model(test_input)
            loaded_output = loaded_model(test_input)
            
        torch.testing.assert_close(
            original_output['logits'],
            loaded_output['logits']
        )