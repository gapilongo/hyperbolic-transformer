from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from model.transformer import HyperbolicTransformer
from data.tokenizer import EnhancedTokenizer


class VisualizationTool:
    """Tools for visualizing model behavior"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 tokenizer: EnhancedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visualize_attention(self,
                          text: str,
                          layer: int = -1,
                          head: Optional[int] = None) -> None:
        """Visualize attention patterns"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Encode text
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt'
        ).to(self.model.device)
        
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get attention weights from specified layer
        attention = outputs['layer_outputs'][layer]['attention']
        
        if head is not None:
            attention = attention[:, head:head+1]
            
        # Average across batch and heads
        attention = attention.mean(dim=(0, 1)).cpu()
        
        # Decode tokens
        tokens = self.tokenizer.decode(inputs['input_ids'][0]).split()
        
        # Plot attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        plt.title(f'Attention Weights (Layer {layer}{"" if head is None else f", Head {head}"})')
        plt.show()
        
    def visualize_hyperbolic_embeddings(self,
                                      texts: List[str],
                                      method: str = 'pca',
                                      n_components: int = 2) -> None:
        """Visualize embeddings in hyperbolic space"""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Get embeddings for texts
        embeddings = []
        for text in texts:
            inputs = self.tokenizer.encode(
                text,
                return_tensors='pt'
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs['last_hidden_state'].mean(dim=1)
                embeddings.append(embedding)
                
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        
        # Dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
            
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Plot embeddings
        plt.figure(figsize=(10, 8))
        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1]
        )
        
        # Add text labels
        for i, text in enumerate(texts):
            plt.annotate(
                text[:20] + '...' if len(text) > 20 else text,
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1])
            )
            
        plt.title(f'Hyperbolic Embeddings ({method.upper()})')
        plt.show()