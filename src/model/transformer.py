
import torch.nn as nn
import torch
from typing import List, Dict, Tuple, Optional
from src.core.attention import HyperbolicGraphAttention
from src.core.config.configurations import ModelConfig
from src.core.hyperbolic import HyperbolicSpace
from src.model.graph import HyperbolicSpace, EnhancedHyperbolicGraph
from src.model.community import CommunityDetector
from src.model.patterns import HierarchicalPatternProcessor


class HyperbolicTransformer(nn.Module):
    """Main model architecture combining all components"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Standardize dimension usage
        dim = config.hidden_size  # Using hidden_size as the standard dimension
        
        # Core components
        self.hyperbolic = HyperbolicSpace(dim=dim)
        self.graph = EnhancedHyperbolicGraph(config)
        self.community_detector = CommunityDetector(config)
        self.pattern_processor = HierarchicalPatternProcessor(config)
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            dim,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            dim
        )
        
        # Processing layers
        self.input_projection = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            HyperbolicGraphAttention(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Verify parameters are registered
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if num_params == 0:
            raise ValueError(
                "Model has no trainable parameters! Check if all components "
                "are properly inheriting from nn.Module and registering parameters."
            )
    
    def _init_weights(self, module):
        """Initialize weights with enhanced stability"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                mlm_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced processing"""
        # Get device once
        device = next(self.parameters()).device
        
        # Move inputs to correct device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
        position_ids = position_ids.to(device) if position_ids is not None else None
        mlm_labels = mlm_labels.to(device) if mlm_labels is not None else None

        # Handle arbitrary input shapes
        input_shape = input_ids.size()
        
        # Determine effective batch size and sequence length
        if len(input_shape) == 1:
            batch_size = 1
            seq_length = input_shape[0]
            input_ids = input_ids.unsqueeze(0)
        elif len(input_shape) == 2:
            batch_size, seq_length = input_shape
        elif len(input_shape) == 3:
            batch_size, num_choices, seq_length = input_shape
            input_ids = input_ids.view(-1, seq_length)
        else:
            raise ValueError(
                f"Input shape {input_shape} is not supported. Expected 1D, 2D, or 3D tensor. "
                f"Got shape: {input_shape}"
            )
            
        # Ensure all optional inputs have correct shapes
        def reshape_input(tensor: Optional[torch.Tensor], target_shape: torch.Size) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
                
            if tensor.shape == target_shape:
                return tensor
                
            # Try to broadcast/reshape the tensor
            if len(tensor.shape) < len(target_shape):
                # Add missing dimensions
                for _ in range(len(target_shape) - len(tensor.shape)):
                    tensor = tensor.unsqueeze(0)
            
            try:
                return tensor.expand(target_shape)
            except RuntimeError:
                try:
                    return tensor.view(target_shape)
                except RuntimeError:
                    raise ValueError(
                        f"Cannot reshape tensor of shape {tensor.shape} to {target_shape}"
                    )
        
        # Get target shapes for each input type
        target_shape = input_ids.shape
        attention_mask = reshape_input(attention_mask, target_shape)
        token_type_ids = reshape_input(token_type_ids, target_shape)
        
        # Handle position IDs specially
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(target_shape)
        else:
            position_ids = reshape_input(position_ids, target_shape)
        
        # MLM labels might have different shape requirements
        if mlm_labels is not None:
            mlm_labels = reshape_input(mlm_labels, target_shape)
        
        # Get embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Project to hyperbolic space
        hyperbolic_embeddings = self.hyperbolic.exp_map(
            torch.zeros_like(embeddings),
            self.input_projection(embeddings)
        )
        
        # Process through attention layers
        layer_outputs = []
        current_embeddings = hyperbolic_embeddings
        
        for layer in self.attention_layers:
            # Update graph structure
            self._update_graph_structure(current_embeddings)
            
            # Apply attention with proper attention mask
            layer_output, attention_weights = layer(
                current_embeddings,
                attention_mask=attention_mask
            )
            
            # Update patterns
            processed_patterns = self.pattern_processor.process_patterns(
                layer_output,
                levels=[0] * batch_size  # Default level
            )
            
            layer_outputs.append({
                'embeddings': layer_output,
                'attention': attention_weights,
                'patterns': processed_patterns
            })
            
            current_embeddings = layer_output
        
        # Final processing
        output_embeddings = self.hyperbolic.log_map(
            torch.zeros_like(current_embeddings),
            current_embeddings
        )
        
        logits = self.output_projection(output_embeddings)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'last_hidden_state': current_embeddings,
            'layer_outputs': layer_outputs,
            'embeddings': embeddings
        }
        
        # Handle MLM if labels are provided
        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size), 
                mlm_labels.view(-1)
            )
            outputs['loss'] = mlm_loss

        return outputs
        
    def _update_graph_structure(self, embeddings: torch.Tensor) -> None:
        """Update graph structure with new embeddings"""
        batch_size, seq_length = embeddings.size()[:2]
        
        # Flatten embeddings for processing
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        
        # Update nodes and edges
        for i in range(len(flat_embeddings)):
            node_id = i
            embedding = flat_embeddings[i]
            
            # Add or update node
            self.graph.add_node(node_id, embedding)
            
            # Add edges to nearby nodes
            for j in range(max(0, i-self.config.max_position_embeddings),
                         min(len(flat_embeddings), i+self.config.max_position_embeddings)):
                if i != j:
                    target_embedding = flat_embeddings[j]
                    similarity = -self.hyperbolic.distance(embedding, target_embedding)
                    
                    if similarity > self.config.edge_importance_threshold:
                        self.graph.add_edge(node_id, j, weight=similarity.item())
                        
        # Update communities periodically
        if self.training and torch.rand(1).item() < 0.1:  # 10% chance
            communities = self.community_detector.detect_communities(self.graph)
            for node_id, community_id in enumerate(communities):
                if node_id in self.graph.nodes:
                    self.graph.nodes[node_id].community_id = community_id