from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import torch
import networkx as nx
from collections import defaultdict, Counter
import time
from src.core.hyperbolic import HyperbolicSpace
from src.core.config.configurations import ModelConfig
from src.core.attention import HyperbolicGraphAttention


@dataclass
class NodeState:
    """Enhanced node state tracking"""
    embedding: torch.Tensor
    community_id: int = -1
    hierarchical_level: int = 0
    incoming_edges: Set[int] = field(default_factory=set)
    outgoing_edges: Set[int] = field(default_factory=set)
    last_update: float = 0.0
    update_count: int = 0
    importance_score: float = 0.0

class EnhancedHyperbolicGraph:
    """Advanced hyperbolic graph with dynamic structure"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.hyperbolic = HyperbolicSpace(dim=config.hidden_size)
        self.graph = nx.DiGraph()
        self.nodes: Dict[int, NodeState] = {}
        self.attention = HyperbolicGraphAttention(config)
        
        # Hierarchical structure
        self.communities = []
        self.hierarchical_levels = defaultdict(int)
        
        # Learning rate scheduler for node updates
        self.base_lr = 0.01
        self.min_lr = 0.001
        self.decay_factor = 0.995
        
        # Edge pruning parameters
        self.max_edges_per_node = 100
        self.edge_importance_threshold = 0.1
        
    def add_node(self, 
                 node_id: int, 
                 embedding: torch.Tensor,
                 community_id: int = -1,
                 level: int = 0) -> None:
        """Add node with enhanced state tracking"""
        if node_id not in self.nodes:
            # Project embedding to hyperbolic space
            embedding = self.hyperbolic.project(embedding)
            
            # Initialize node state
            self.nodes[node_id] = NodeState(
                embedding=embedding,
                community_id=community_id,
                hierarchical_level=level,
                importance_score=1.0
            )
            
            self.graph.add_node(node_id)
        else:
            # Update existing node
            current_state = self.nodes[node_id]
            lr = max(self.min_lr, 
                    self.base_lr * (self.decay_factor ** current_state.update_count))
            
            # Interpolate in hyperbolic space
            new_embedding = self.hyperbolic.exp_map(
                current_state.embedding,
                lr * self.hyperbolic.log_map(current_state.embedding, embedding)
            )
            
            current_state.embedding = new_embedding
            current_state.update_count += 1
            current_state.last_update = time.time()
    
    def add_edge(self, 
                 source: int, 
                 target: int,
                 weight: float = 1.0) -> None:
        """Add edge with importance scoring"""
        if source not in self.nodes or target not in self.nodes:
            return
            
        source_state = self.nodes[source]
        target_state = self.nodes[target]
        
        # Compute edge importance
        edge_features = torch.cat([
            source_state.embedding,
            target_state.embedding
        ], dim=-1)
        
        importance = self.attention.edge_importance(edge_features).item()
        
        if importance > self.edge_importance_threshold:
            # Check edge count limit
            if len(source_state.outgoing_edges) >= self.max_edges_per_node:
                # Remove least important edge
                min_importance = float('inf')
                edge_to_remove = None
                
                for target_id in source_state.outgoing_edges:
                    edge_data = self.graph.get_edge_data(source, target_id)
                    if edge_data['importance'] < min_importance:
                        min_importance = edge_data['importance']
                        edge_to_remove = target_id
                
                if edge_to_remove is not None and min_importance < importance:
                    self.remove_edge(source, edge_to_remove)
                else:
                    return
            
            # Add edge with metadata
            self.graph.add_edge(
                source, target,
                weight=weight,
                importance=importance
            )
            
            source_state.outgoing_edges.add(target)
            target_state.incoming_edges.add(source)
    
    def remove_edge(self, source: int, target: int) -> None:
        """Remove edge and update node states"""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self.nodes[source].outgoing_edges.remove(target)
            self.nodes[target].incoming_edges.remove(source)
    
    def update_node_importance(self) -> None:
        """Update node importance scores based on connectivity"""
        for node_id, state in self.nodes.items():
            # Compute importance based on incoming and outgoing connections
            in_degree = len(state.incoming_edges)
            out_degree = len(state.outgoing_edges)
            
            # PageRank-like importance score
            state.importance_score = 0.15 + 0.85 * (
                in_degree / (max(1, self.graph.number_of_nodes() - 1))
            )
            
            # Adjust by hierarchical level
            state.importance_score *= (1 + 0.1 * state.hierarchical_level)