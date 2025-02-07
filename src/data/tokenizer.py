from typing import List, Dict, Optional, Union, Any
import torch
import os
import json
import sentencepiece as spm
import re
from src.core.config.configurations import TokenizerConfig

class EnhancedTokenizer:
    """Advanced tokenizer with subword units"""
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.sp_model = None
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Add special tokens
        self.special_tokens = {
            'pad_token': config.pad_token,
            'unk_token': config.unk_token,
            'cls_token': config.cls_token,
            'sep_token': config.sep_token,
            'mask_token': config.mask_token
        }
        
    def train(self, texts: List[str], output_path: str):
        """Train tokenizer on texts"""
        # Write texts to temporary file
        with open('temp_train.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
                
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input='temp_train.txt',
            model_prefix=output_path,
            vocab_size=self.config.vocab_size - len(self.special_tokens),
            character_coverage=0.9995,
            model_type='bpe',
            # min_frequency=self.config.min_frequency,
            pad_id=-1,
            unk_id=0
        )
        
        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{output_path}.model")
        
        # Build vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        sp_vocab_size = len(self.sp_model)
        
        for idx in range(sp_vocab_size):
            token = self.sp_model.id_to_piece(idx)
            self.vocab[token] = idx + len(self.special_tokens)
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def encode(self,
              text: Union[str, List[str]],
              max_length: Optional[int] = None,
              padding: bool = True,
              truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Encode text to token ids"""
        if isinstance(text, str):
            text = [text]
            
        max_length = max_length or self.config.max_length
        
        # Initialize outputs
        input_ids = []
        attention_mask = []
        
        for t in text:
            # Add CLS token
            tokens = [self.special_tokens['cls_token']]
            
            # Tokenize text
            sp_tokens = self.sp_model.encode_as_pieces(t)
            tokens.extend(sp_tokens)
            
            # Add SEP token
            tokens.append(self.special_tokens['sep_token'])
            
            # Convert to ids
            ids = [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']])
                  for token in tokens]
            
            # Truncate if needed
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.vocab[self.special_tokens['sep_token']]]
                
            # Create attention mask
            mask = [1] * len(ids)
            
            # Pad if needed
            if padding and len(ids) < max_length:
                pad_length = max_length - len(ids)
                ids.extend([self.vocab[self.special_tokens['pad_token']]] * pad_length)
                mask.extend([0] * pad_length)
                
            input_ids.append(ids)
            attention_mask.append(mask)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token ids back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Remove special tokens
        cleaned_ids = []
        for id in token_ids:
            token = self.inverse_vocab.get(id)
            if token not in self.special_tokens.values():
                cleaned_ids.append(id)
                
        # Convert to SentencePiece ids
        sp_ids = [id - len(self.special_tokens) for id in cleaned_ids 
                 if id >= len(self.special_tokens)]
        
        # Decode with SentencePiece
        return self.sp_model.decode(sp_ids)
    
    def save(self, path: str):
        """Save tokenizer files"""
        os.makedirs(path, exist_ok=True)
        
        # Save SentencePiece model
        sp_path = os.path.join(path, "tokenizer.model")
        if self.sp_model:
            self.sp_model.save(sp_path)
            
        # Save vocabulary and config
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': self.special_tokens
            }, f, indent=2)
            
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EnhancedTokenizer':
        """Load tokenizer from saved files"""
        # Load config
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TokenizerConfig(**config_dict)
        
        # Initialize tokenizer
        tokenizer = cls(config)
        
        # Load vocabulary
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        tokenizer.vocab = vocab_dict['vocab']
        tokenizer.special_tokens = vocab_dict['special_tokens']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        # Load SentencePiece model
        sp_path = os.path.join(path, "tokenizer.model")
        if os.path.exists(sp_path):
            tokenizer.sp_model = spm.SentencePieceProcessor()
            tokenizer.sp_model.load(sp_path)
            
        return tokenizer
    
    def token_to_id(self, token: str) -> int:
        """Convert token to id"""
        return self.vocab.get(token, self.vocab[self.special_tokens['unk_token']])
    
    def id_to_token(self, id: int) -> str:
        """Convert id to token"""
        return self.inverse_vocab.get(id, self.special_tokens['unk_token'])

class TextPreprocessor:
    """Advanced text preprocessing pipeline"""
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 fix_unicode: bool = True,
                 normalize_whitespace: bool = True):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.fix_unicode = fix_unicode
        self.normalize_whitespace = normalize_whitespace
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'[\w\.-]+@[\w\.-]+\.\w+'
        )
        
    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline"""
        if not text or not isinstance(text, str):
            return ""
            
        # Fix unicode
        if self.fix_unicode:
            text = self._fix_unicode(text)
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Normalize whitespace
        if self.normalize_whitespace:
            text = ' '.join(text.split())
            
        return text.strip()
    
    def _fix_unicode(self, text: str) -> str:
        """Fix common unicode issues"""
        # Replace smart quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace other common characters
        replacements = {
            '…': '...',
            '–': '-',
            '—': '-',
            '•': '*',
            '\u200b': '',  # Zero width space
            '\ufeff': ''   # Byte order mark
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
