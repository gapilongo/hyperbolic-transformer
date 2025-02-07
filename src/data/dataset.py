from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Optional, Union, Any,Tuple
from tqdm import tqdm
import numpy as np
import random
from src.data.tokenizer import EnhancedTokenizer



class TextDataset(Dataset):
    """Enhanced dataset for text processing"""
    def __init__(self, 
                 texts: List[str],
                 tokenizer: EnhancedTokenizer,
                 max_length: Optional[int] = None,
                 is_training: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.config.max_length
        self.is_training = is_training
        
        # Pre-tokenize all texts for efficiency
        self.encoded_data = []
        for text in tqdm(texts, desc="Encoding texts"):
            encoded = self.tokenize_and_process(text)
            self.encoded_data.append(encoded)
            
    def tokenize_and_process(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and process text with advanced features"""
        # Basic tokenization
        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Ensure all tensors are on CPU initially
        encoding = {
            k: torch.tensor(v, dtype=torch.long, device='cpu')
            if isinstance(v, (list, np.ndarray)) else v
            for k, v in encoding.items()
        }
        
        if self.is_training:
            # Add masked language modeling targets
            mlm_inputs, mlm_labels = self.create_mlm_inputs(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            encoding['input_ids'] = mlm_inputs
            encoding['mlm_labels'] = mlm_labels
                
            # Add next sentence prediction if possible
            if len(text.split('\n')) > 1:
                nsp_label = self.create_nsp_inputs(encoding)
                encoding['nsp_label'] = torch.tensor(nsp_label, dtype=torch.long, device='cpu')
        
        return encoding
    
    def create_mlm_inputs(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        mlm_probability: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create inputs for masked language modeling"""
        mlm_inputs = input_ids.clone()
        mlm_labels = input_ids.clone()
        
        # Get the list of special token IDs
        special_token_ids = set(self.tokenizer.special_tokens.values())
        
        # Create special tokens mask for the entire sequence at once
        special_tokens_mask = torch.tensor([
            int(id in special_token_ids) 
            for id in range(len(self.tokenizer.vocab))
        ], dtype=torch.bool)
        
        # Create probability matrix
        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        
        # Apply masks:
        # 1. Mask tokens that are padding (attention_mask == 0)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        # 2. Mask special tokens by indexing into our pre-computed mask
        if len(input_ids.shape) == 1:
            probability_matrix.masked_fill_(special_tokens_mask[input_ids], value=0.0)
        else:
            # Handle batched input
            for i in range(input_ids.shape[0]):
                probability_matrix[i].masked_fill_(special_tokens_mask[input_ids[i]], value=0.0)
        
        # Generate masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        
        # Get mask token ID
        mask_token_id = self.tokenizer.vocab[self.tokenizer.special_tokens['mask_token']]
        mlm_inputs[indices_replaced] = mask_token_id
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), input_ids.shape, dtype=torch.long)
        mlm_inputs[indices_random] = random_words[indices_random]
        
        return mlm_inputs, mlm_labels
        

    def create_nsp_inputs(self, encoding: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create inputs for next sentence prediction"""
        # 50% of the time, we create a random next sentence
        if random.random() < 0.5:
            return torch.tensor(1)  # Not actual next sentence
        return torch.tensor(0)  # Actual next sentence
    
    def __len__(self) -> int:
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_data[idx]

class DynamicBatchSampler:
    """Dynamic batch sampler for efficient processing"""
    def __init__(self,
                 dataset: TextDataset,
                 batch_size: int,
                 max_tokens: int = 4096,
                 shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        
        # Group similar length sequences together
        self.sequence_lengths = [
            len(data['input_ids']) for data in dataset.encoded_data
        ]
        self.indices = list(range(len(dataset)))
        
        if shuffle:
            random.shuffle(self.indices)
            
    def __iter__(self):
        """Yield batches of indices"""
        if self.shuffle:
            random.shuffle(self.indices)
            
        current_batch = []
        current_length = 0
        
        for idx in self.indices:
            seq_length = self.sequence_lengths[idx]
            batch_tokens = (len(current_batch) + 1) * seq_length
            
            if batch_tokens > self.max_tokens and current_batch:
                yield current_batch
                current_batch = []
                current_length = 0
                
            current_batch.append(idx)
            current_length = max(current_length, seq_length)
            
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []
                current_length = 0
                
        if current_batch:
            yield current_batch
            
    def __len__(self):
        """Approximate number of batches"""
        if self.max_tokens == float('inf'):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            
        total_tokens = sum(self.sequence_lengths)
        return total_tokens // self.max_tokens + 1

def create_dataloader(dataset: TextDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create dataloader with dynamic batching"""
    sampler = DynamicBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    def collate_fn(batch):
        """Custom collate function for batching"""
        # Combine all tensors in the batch
        batch_dict = {
            key: torch.stack([b[key] for b in batch])
            for key in batch[0].keys()
        }
        
        return batch_dict
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )