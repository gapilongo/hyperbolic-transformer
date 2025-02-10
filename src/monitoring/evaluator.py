import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from src.core.config.configurations import ModelConfig
from src.model.transformer import HyperbolicTransformer
from src.data.tokenizer import EnhancedTokenizer


class ModelEvaluator:
    """Comprehensive model evaluation"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer,
                 config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        
    def evaluate(self,
                eval_dataloader: DataLoader,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Run full evaluation"""
        self.model.eval()
        metrics = metrics or ['perplexity', 'accuracy', 'bleu', 'rouge']
        
        # Initialize metrics
        results = defaultdict(float)
        total_examples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(**batch)
                
                # Update metrics
                batch_metrics = self._compute_batch_metrics(
                    outputs,
                    batch,
                    metrics
                )
                
                for key, value in batch_metrics.items():
                    results[key] += value * len(batch['input_ids'])
                    
                total_examples += len(batch['input_ids'])
        
        # Average metrics
        for key in results:
            results[key] /= total_examples
            self.metrics[key].append(results[key])
            
            # Update best metrics
            if key not in self.best_metrics or results[key] > self.best_metrics[key]:
                self.best_metrics[key] = results[key]
        
        return dict(results)
    
    def _compute_batch_metrics(self,
                             outputs: Dict[str, torch.Tensor],
                             batch: Dict[str, torch.Tensor],
                             metrics: List[str]) -> Dict[str, float]:
        """Compute metrics for a single batch"""
        results = {}
        
        if 'perplexity' in metrics:
            results['perplexity'] = self._compute_perplexity(
                outputs['logits'],
                batch['input_ids']
            )
            
        if 'accuracy' in metrics:
            results['accuracy'] = self._compute_accuracy(
                outputs['logits'],
                batch['input_ids']
            )
            
        if 'bleu' in metrics:
            results['bleu'] = self._compute_bleu(
                outputs['logits'],
                batch['input_ids']
            )
            
        if 'rouge' in metrics:
            results['rouge'] = self._compute_rouge(
                outputs['logits'],
                batch['input_ids']
            )
            
        return results
    
    def _compute_perplexity(self,
                          logits: torch.Tensor,
                          labels: torch.Tensor) -> float:
        """Compute perplexity score"""
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.token_to_id(self.tokenizer.special_tokens['pad_token'])
        )
        return torch.exp(loss).item()
    
    def _compute_accuracy(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor) -> float:
        """Compute token prediction accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != self.tokenizer.token_to_id(self.tokenizer.special_tokens['pad_token'])
        correct = (predictions == labels) & mask
        return correct.float().mean().item()
    
    def _compute_bleu(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor) -> float:
        """Compute BLEU score"""
        predictions = torch.argmax(logits, dim=-1)
        
        # Convert to text
        pred_texts = [self.tokenizer.decode(pred) for pred in predictions]
        label_texts = [self.tokenizer.decode(label) for label in labels]
        
        # Compute BLEU score
        bleu_score = self._calculate_bleu(pred_texts, label_texts)
        return bleu_score
    
    def _compute_rouge(self,
                      logits: torch.Tensor,
                      labels: torch.Tensor) -> Dict[str, float]:
        """Compute ROUGE scores"""
        predictions = torch.argmax(logits, dim=-1)
        
        # Convert to text
        pred_texts = [self.tokenizer.decode(pred) for pred in predictions]
        label_texts = [self.tokenizer.decode(label) for label in labels]
        
        # Compute ROUGE scores
        rouge_scores = self._calculate_rouge(pred_texts, label_texts)
        return rouge_scores
    
    def _calculate_bleu(self,
                       predictions: List[str],
                       references: List[str]) -> float:
        """Calculate BLEU score"""
        from nltk.translate.bleu_score import corpus_bleu
        
        # Tokenize predictions and references
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        return corpus_bleu(ref_tokens, pred_tokens)
    
    def _calculate_rouge(self,
                        predictions: List[str],
                        references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        scores = defaultdict(float)
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(pred, ref)
            for key, value in score.items():
                scores[key] += value.fmeasure
                
        # Average scores
        for key in scores:
            scores[key] /= len(predictions)
            
        return dict(scores)
    
    def analyze_attention_patterns(self,
                                 dataloader: DataLoader) -> Dict[str, Any]:
        """Analyze attention patterns"""
        self.model.eval()
        attention_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing attention"):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Get model outputs with attention
                outputs = self.model(**batch)
                
                # Analyze attention patterns for each layer
                for layer_idx, layer_output in enumerate(outputs['layer_outputs']):
                    attention_weights = layer_output['attention']
                    
                    # Compute attention statistics
                    stats = self._analyze_layer_attention(
                        attention_weights,
                        batch['attention_mask']
                    )
                    
                    for key, value in stats.items():
                        attention_stats[f"layer_{layer_idx}_{key}"].append(value)
        
        # Average statistics across batches
        final_stats = {}
        for key, values in attention_stats.items():
            final_stats[key] = torch.stack(values).mean().item()
            
        return final_stats
    
    def _analyze_layer_attention(self,
                               attention_weights: torch.Tensor,
                               attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns in a single layer"""
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        
        stats = {}
        
        # Average attention
        mean_attention = attention_weights.mean(dim=1)  # Average across heads
        
        # Apply mask
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        masked_attention = mean_attention * mask
        
        # Compute statistics
        stats['attention_entropy'] = self._compute_attention_entropy(masked_attention)
        stats['attention_sparsity'] = self._compute_attention_sparsity(masked_attention)
        stats['attention_coverage'] = self._compute_attention_coverage(masked_attention)
        
        return stats
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions"""
        # Avoid log(0)
        eps = 1e-8
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
        return entropy.mean()
    
    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of attention weights"""
        threshold = 0.1
        sparsity = (attention_weights < threshold).float().mean()
        return sparsity
    
    def _compute_attention_coverage(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute coverage of attention weights"""
        threshold = 0.05
        coverage = (attention_weights > threshold).float().mean()
        return coverage