from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
from attribution import AttributionComputer
from error import ErrorAnalyzer
from patterns import PatternAnalyzer
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer
from data.tokenizer import EnhancedTokenizer


class ModelInterpreter:
    """Advanced model interpretation and analysis"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer,
                 config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize interpretation components
        self.pattern_analyzer = PatternAnalyzer(model, config)
        self.attribution_computer = AttributionComputer(model, tokenizer)
        self.error_analyzer = ErrorAnalyzer(model, tokenizer)
        
    def interpret_prediction(self,
                           text: str,
                           target: Optional[str] = None,
                           return_viz: bool = False) -> Dict[str, Any]:
        """Comprehensive interpretation of model prediction"""
        # Encode input
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt'
        ).to(self.model.device)
        
        # Get model outputs with gradients
        outputs = self.get_outputs_with_gradients(inputs, target)
        
        # Compute various interpretations
        interpretations = {
            'token_attributions': self.attribution_computer.compute_attributions(
                inputs, outputs
            ),
            'attention_patterns': self.pattern_analyzer.analyze_patterns(outputs),
            'error_analysis': self.error_analyzer.analyze_errors(
                inputs, outputs, target
            ) if target else None,
            'decision_explanation': self.explain_decision(outputs)
        }
        
        if return_viz:
            interpretations['visualizations'] = self.create_interpretation_viz(
                text, interpretations
            )
            
        return interpretations
    
    def get_outputs_with_gradients(self,
                                 inputs: Dict[str, torch.Tensor],
                                 target: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get model outputs with gradient tracking"""
        # Enable gradient tracking
        for tensor in inputs.values():
            tensor.requires_grad_(True)
            
        # Forward pass
        outputs = self.model(**inputs)
        
        if target:
            # Compute loss for target
            target_ids = self.tokenizer.encode(target, return_tensors='pt').to(self.model.device)
            loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                target_ids.view(-1)
            )
            loss.backward()
            
        return outputs
    
    def explain_decision(self, outputs: Dict[str, torch.Tensor]) -> str:
        """Generate natural language explanation of model's decision"""
        # Get top predictions
        logits = outputs['logits'][:, -1]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        
        # Get attention patterns
        attention = outputs['layer_outputs'][-1]['attention']
        
        # Generate explanation
        explanation = self._generate_explanation(
            top_indices, top_probs, attention
        )
        
        return explanation
    
    def _generate_explanation(self,
                            indices: torch.Tensor,
                            probs: torch.Tensor,
                            attention: torch.Tensor) -> str:
        """Generate detailed explanation of model's reasoning"""
        explanation_parts = []
        
        # Explain top predictions
        explanation_parts.append("Top predictions:")
        for idx, prob in zip(indices[0], probs[0]):
            token = self.tokenizer.decode([idx])
            explanation_parts.append(f"- {token}: {prob:.2%}")
            
        # Analyze attention patterns
        important_tokens = self._get_important_attention_tokens(attention)
        
        if important_tokens:
            explanation_parts.append("\nKey attention points:")
            for token, score in important_tokens:
                explanation_parts.append(f"- {token}: {score:.2f} attention weight")
                
        return "\n".join(explanation_parts)
    
    def _get_important_attention_tokens(self,
                                     attention: torch.Tensor,
                                     threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Extract tokens with significant attention weights"""
        # Average attention across heads
        mean_attention = attention.mean(dim=1)
        
        # Get important token indices
        important_indices = torch.where(mean_attention > threshold)
        
        # Convert to tokens and scores
        important_tokens = []
        for idx, score in zip(important_indices[0], mean_attention[important_indices]):
            token = self.tokenizer.decode([idx])
            important_tokens.append((token, score.item()))
            
        return sorted(important_tokens, key=lambda x: x[1], reverse=True)