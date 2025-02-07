import torch
import onnx
import tensorrt as trt
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path
from torch.jit import trace
from core.config.configurations import ServingConfig
from data.tokenizer import EnhancedTokenizer
from model.transformer import HyperbolicTransformer

class ModelExporter:
    """Export model for serving"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer,
                 config: ServingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Ensure model is in eval mode
        self.model.eval()
        
    def export_torchscript(self, path: str):
        """Export model to TorchScript format"""
        # Create example inputs
        example_inputs = self._create_example_inputs()
        
        # Trace model
        traced_model = trace(
            self.model,
            example_inputs=(example_inputs,)
        )
        
        # Save model
        torch.jit.save(traced_model, path)
        
        # Export config and tokenizer
        self._save_serving_config(path)
        self._save_tokenizer(path)
        
    def export_onnx(self, path: str):
        """Export model to ONNX format"""
        example_inputs = self._create_example_inputs()
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            example_inputs,
            path,
            opset_version=13,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        
        # Export config and tokenizer
        self._save_serving_config(path)
        self._save_tokenizer(path)
        
    def export_tensorrt(self, path: str):
        """Export model to TensorRT format"""
        if not self.config.use_tensorrt:
            raise ValueError("TensorRT export not enabled in config")
            
        # First export to ONNX
        onnx_path = path.replace('.trt', '.onnx')
        self.export_onnx(onnx_path)
        
        # Convert to TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
            
        # Create optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input_ids',
            min=(1, 1),
            opt=(self.config.batch_size, self.config.max_sequence_length),
            max=(self.config.batch_size * 2, self.config.max_sequence_length)
        )
        profile.set_shape(
            'attention_mask',
            min=(1, 1),
            opt=(self.config.batch_size, self.config.max_sequence_length),
            max=(self.config.batch_size * 2, self.config.max_sequence_length)
        )
        
        # Build engine
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        config.max_workspace_size = 1 << 30  # 1GB
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(path, 'wb') as f:
            f.write(engine.serialize())
            
        # Export config and tokenizer
        self._save_serving_config(path)
        self._save_tokenizer(path)
        
    def _create_example_inputs(self) -> Dict[str, torch.Tensor]:
        """Create example inputs for tracing"""
        return {
            'input_ids': torch.zeros(
                (self.config.batch_size, self.config.max_sequence_length),
                dtype=torch.long
            ),
            'attention_mask': torch.ones(
                (self.config.batch_size, self.config.max_sequence_length),
                dtype=torch.long
            )
        }
        
    def _save_serving_config(self, model_path: str):
        """Save serving configuration"""
        config_path = Path(model_path).parent / 'serving_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
            
    def _save_tokenizer(self, model_path: str):
        """Save tokenizer files"""
        tokenizer_path = Path(model_path).parent / 'tokenizer'
        self.tokenizer.save(str(tokenizer_path))