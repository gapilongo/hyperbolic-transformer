from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import json
from src.data.dataset import create_dataloader,TextDataset
from src.data.tokenizer import EnhancedTokenizer, TextPreprocessor


class DataProcessor:
    """Main data processing pipeline"""
    def __init__(self,
                 tokenizer: EnhancedTokenizer,
                 preprocessor: TextPreprocessor,
                 max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length or tokenizer.config.max_length
        
    def process_file(self, file_path: Union[str, Path]) -> List[str]:
        """Process a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read file based on extension
        if file_path.suffix == '.txt':
            texts = self._process_text_file(file_path)
        elif file_path.suffix == '.json':
            texts = self._process_json_file(file_path)
        elif file_path.suffix in ['.csv', '.tsv']:
            texts = self._process_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        return texts
    
    def _process_text_file(self, file_path: Path) -> List[str]:
        """Process plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split into manageable chunks
        chunks = self._split_into_chunks(text)
        
        # Preprocess each chunk
        processed_chunks = [
            self.preprocessor.preprocess(chunk)
            for chunk in chunks
        ]
        
        return [chunk for chunk in processed_chunks if chunk]
    
    def _process_json_file(self, file_path: Path) -> List[str]:
        """Process JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract text fields recursively
        texts = []
        self._extract_text_fields(data, texts)
        
        # Preprocess each text
        processed_texts = [
            self.preprocessor.preprocess(text)
            for text in texts
        ]
        
        return [text for text in processed_texts if text]
    
    def _process_csv_file(self, file_path: Path) -> List[str]:
        """Process CSV/TSV file"""
        import pandas as pd
        
        # Detect delimiter
        delimiter = ',' if file_path.suffix == '.csv' else '\t'
        
        # Read file
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Extract text from all string columns
        texts = []
        for column in df.select_dtypes(include=['object']):
            texts.extend(df[column].dropna().tolist())
            
        # Preprocess each text
        processed_texts = [
            self.preprocessor.preprocess(str(text))
            for text in texts
        ]
        
        return [text for text in processed_texts if text]
    
    def _extract_text_fields(self, data: Any, texts: List[str]):
        """Recursively extract text fields from JSON structure"""
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, list):
            for item in data:
                self._extract_text_fields(item, texts)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_text_fields(value, texts)
                
    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def prepare_training_data(self,
                            texts: List[str],
                            batch_size: int = 32,
                            shuffle: bool = True) -> DataLoader:
        """Prepare data for training"""
        # Create dataset
        dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_training=True
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader