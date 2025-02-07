from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from core.config.configurations import ModelConfig
from model.transformer import HyperbolicTransformer
from data.tokenizer import EnhancedTokenizer
from utils.memory import LRUCache, PatternMemory

class TextGenerator:
    """Advanced text generation with hyperbolic guidance"""
    def __init__(self, 
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer,
                 config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Generation settings
        self.max_length = 512
        self.min_length = 10
        self.top_k = 50
        self.top_p = 0.95
        self.temperature = 0.7
        self.repetition_penalty = 1.2
        self.length_penalty = 1.0
        
        # Context tracking
        self.context_cache = LRUCache(maxsize=1000)
        self.pattern_memory = PatternMemory(config)
        
    def generate(self,
                prompt: str,
                max_length: Optional[int] = None,
                num_return_sequences: int = 1,
                **kwargs) -> List[str]:
        """Generate text with advanced controls"""
        # Update generation settings from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        max_length = max_length or self.max_length
        
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        )["input_ids"].to(self.model.device)
        
        # Generate sequences
        outputs = []
        for _ in range(num_return_sequences):
            output = self._generate_sequence(
                input_ids,
                max_length=max_length
            )
            outputs.append(output)
            
        return outputs
    
    def _generate_sequence(self,
                        input_ids: torch.Tensor,
                        max_length: int) -> str:
        """Generate a single sequence"""
        # Initialize sequence with input
        current_ids = input_ids.clone()
        
        # Track generated patterns
        generated_patterns = []
        
        # Generation loop
        while current_ids.size(1) < max_length:
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(current_ids)
                
            # Extract next token probabilities
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / self.temperature
            
            # Apply repetition penalty
            for seq_idx in range(current_ids.size(0)):
                for prev_token in set(current_ids[seq_idx].tolist()):
                    next_token_logits[seq_idx, prev_token] /= self.repetition_penalty
            
            # Filter with top-k
            if self.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Filter with top-p (nucleus sampling)
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply hyperbolic guidance
            next_token_logits = self._apply_hyperbolic_guidance(
                next_token_logits,
                outputs,
                generated_patterns
            )
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # Update patterns
            if "patterns" in outputs:
                generated_patterns.append(outputs["patterns"])
            
            # Check for completion
            if next_token.item() == self.tokenizer.token_to_id(self.tokenizer.special_tokens["sep_token"]):
                break
        
        # Decode generated sequence
        generated_text = self.tokenizer.decode(current_ids[0])
        
        return generated_text
    
    def _apply_hyperbolic_guidance(self,
                                logits: torch.Tensor,
                                outputs: Dict[str, torch.Tensor],
                                generated_patterns: List[torch.Tensor]) -> torch.Tensor:
        """Apply hyperbolic space guidance to logits"""
        # Get current embedding in hyperbolic space
        current_embedding = outputs["last_hidden_state"][:, -1]
        
        # Get pattern predictions
        pattern_logits = self.pattern_memory.predict_next(
            current_embedding,
            generated_patterns
        )
        
        # Combine with token logits
        if pattern_logits is not None:
            combined_logits = (
                0.7 * logits +
                0.3 * pattern_logits
            )
        else:
            combined_logits = logits
            
        return combined_logits