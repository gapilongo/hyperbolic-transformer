from collections import defaultdict
import numpy as np
from sklearn.cluster import SpectralClustering
from typing import List, Dict, Tuple, Optional
import math
import torch
from src.model.graph import EnhancedHyperbolicGraph, NodeState
from src.core.config.configurations import ModelConfig

class CommunityDetector:
    """Enhanced community detection with hierarchical structure"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.min_community_size = 5
        self.max_hierarchy_levels = 5
        self.similarity_threshold = 0.5
        
        # Community stability tracking
        self.community_history = defaultdict(list)
        self.stability_threshold = 3
        
        # Hierarchical structure
        self.level_communities = defaultdict(dict)
        self.community_centers = {}
        
    def detect_communities(self, 
                         graph: EnhancedHyperbolicGraph,
                         num_communities: Optional[int] = None) -> List[int]:
        """Detect communities with stability checking"""
        if num_communities is None:
            num_communities = self.config.num_communities
            
        num_nodes = len(graph.nodes)
        if num_nodes < 2:
            return [0] * num_nodes
            
        # Build similarity matrix
        similarity_matrix = np.zeros((num_nodes, num_nodes))
        node_indices = list(graph.nodes.keys())
        
        for i, node_i in enumerate(node_indices):
            for j, node_j in enumerate(node_indices):
                if i < j:
                    # Compute similarity score
                    similarity = self._compute_node_similarity(
                        graph.nodes[node_i],
                        graph.nodes[node_j],
                        graph
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                    
        # Normalize similarity matrix
        degree = np.sum(similarity_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        degree_inv_sqrt[~np.isfinite(degree_inv_sqrt)] = 0
        
        laplacian = np.eye(num_nodes) - (
            similarity_matrix * degree_inv_sqrt[:, np.newaxis] * 
            degree_inv_sqrt[np.newaxis, :]
        )
        
        # Adjust number of communities based on eigenvalue analysis
        if num_communities is None:
            eigenvalues = np.linalg.eigvalsh(laplacian)
            gaps = np.diff(eigenvalues)
            num_communities = np.argmax(gaps[:int(num_nodes/2)]) + 1
            num_communities = max(2, min(num_communities, int(num_nodes/4)))
            
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=num_communities,
            affinity='precomputed',
            random_state=42
        ).fit(similarity_matrix)
        
        communities = clustering.labels_
        
        # Update community history and check stability
        for node_idx, community_id in enumerate(communities):
            node_id = node_indices[node_idx]
            self.community_history[node_id].append(community_id)
            
            # Keep only recent history
            if len(self.community_history[node_id]) > self.stability_threshold:
                self.community_history[node_id].pop(0)
        
        # Adjust unstable assignments
        for node_idx, community_id in enumerate(communities):
            node_id = node_indices[node_idx]
            history = self.community_history[node_id]
            
            if len(history) >= self.stability_threshold:
                if history.count(community_id) < self.stability_threshold - 1:
                    # Find most common stable assignment
                    common_community = max(set(history), key=history.count)
                    communities[node_idx] = common_community
        
        # Update community centers
        self._update_community_centers(graph, communities, node_indices)
        
        return communities
    
    def _compute_node_similarity(self,
                               node1: NodeState,
                               node2: NodeState,
                               graph: EnhancedHyperbolicGraph) -> float:
        """Compute comprehensive node similarity"""
        # Embedding similarity in hyperbolic space
        emb_similarity = graph.hyperbolic.distance(
            node1.embedding,
            node2.embedding
        ).item()
        emb_similarity = 1 / (1 + emb_similarity)  # Convert distance to similarity
        
        # Structural similarity
        common_neighbors = (
            len(node1.incoming_edges & node2.incoming_edges) +
            len(node1.outgoing_edges & node2.outgoing_edges)
        )
        total_neighbors = (
            len(node1.incoming_edges | node2.incoming_edges) +
            len(node1.outgoing_edges | node2.outgoing_edges)
        )
        structural_similarity = common_neighbors / max(1, total_neighbors)
        
        # Hierarchical similarity
        level_diff = abs(node1.hierarchical_level - node2.hierarchical_level)
        hierarchical_similarity = math.exp(-level_diff)
        
        # Combine similarities with weights
        similarity = (
            0.5 * emb_similarity +
            0.3 * structural_similarity +
            0.2 * hierarchical_similarity
        )
        
        return similarity
    
    def _update_community_centers(self,
                                graph: EnhancedHyperbolicGraph,
                                communities: List[int],
                                node_indices: List[int]) -> None:
        """Update community centers using Fréchet mean in hyperbolic space"""
        community_members = defaultdict(list)
        
        # Group nodes by community
        for node_idx, community_id in enumerate(communities):
            node_id = node_indices[node_idx]
            community_members[community_id].append(node_id)
        
        # Update centers for each community
        for community_id, members in community_members.items():
            if len(members) < 1:
                continue
                
            # Initialize center as first member
            center = graph.nodes[members[0]].embedding.clone()
            
            # Iteratively update center using Riemannian gradient descent
            max_iter = 50
            lr = 0.1
            prev_center = None
            
            for _ in range(max_iter):
                if prev_center is not None:
                    # Check convergence
                    dist = graph.hyperbolic.distance(center, prev_center)
                    if dist < 1e-6:
                        break
                        
                prev_center = center.clone()
                
                # Compute Riemannian gradient
                gradient = torch.zeros_like(center)
                for node_id in members:
                    node_embedding = graph.nodes[node_id].embedding
                    gradient += graph.hyperbolic.log_map(center, node_embedding)
                
                gradient /= len(members)
                
                # Update center
                center = graph.hyperbolic.exp_map(center, lr * gradient)
            
            self.community_centers[community_id] = center
    
    def get_hierarchical_structure(self, 
                                 graph: EnhancedHyperbolicGraph,
                                 max_levels: Optional[int] = None) -> Dict[int, List[int]]:
        """Get hierarchical community structure"""
        if max_levels is None:
            max_levels = self.max_hierarchy_levels
            
        hierarchy = {}
        current_communities = graph.communities
        
        for level in range(max_levels):
            hierarchy[level] = current_communities
            
            if level < max_levels - 1:
                # Create super-communities for next level
                super_communities = self._create_super_communities(
                    graph,
                    current_communities
                )
                current_communities = super_communities
        
        return hierarchy
    
    def _create_super_communities(self,
                                graph: EnhancedHyperbolicGraph,
                                communities: List[int]) -> List[int]:
        """Create higher-level communities by clustering current communities"""
        # Build community similarity matrix
        unique_communities = sorted(set(communities))
        num_communities = len(unique_communities)
        
        similarity_matrix = np.zeros((num_communities, num_communities))
        
        for i, comm1 in enumerate(unique_communities):
            for j, comm2 in enumerate(unique_communities):
                if i < j:
                    similarity = self._compute_community_similarity(
                        graph,
                        comm1,
                        comm2,
                        communities
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Cluster communities
        num_super_communities = max(2, num_communities // 2)
        clustering = SpectralClustering(
            n_clusters=num_super_communities,
            affinity='precomputed',
            random_state=42
        ).fit(similarity_matrix)
        
        # Map original communities to super-communities
        community_to_super = {
            comm: super_comm
            for comm, super_comm in zip(unique_communities, clustering.labels_)
        }
        
        # Assign super-communities to nodes
        super_communities = [
            community_to_super[comm]
            for comm in communities
        ]
        
        return super_communities
    
    def _compute_community_similarity(self,
                                   graph: EnhancedHyperbolicGraph,
                                   comm1: int,
                                   comm2: int,
                                   communities: List[int]) -> float:
        """Compute similarity between two communities"""
        if comm1 not in self.community_centers or comm2 not in self.community_centers:
            return 0.0
            
        # Center similarity
        center_similarity = 1 / (1 + graph.hyperbolic.distance(
            self.community_centers[comm1],
            self.community_centers[comm2]
        ).item())
        
        # Connectivity similarity
        comm1_nodes = {i for i, c in enumerate(communities) if c == comm1}
        comm2_nodes = {i for i, c in enumerate(communities) if c == comm2}
        
        connections = 0
        for node1 in comm1_nodes:
            for node2 in comm2_nodes:
                if graph.graph.has_edge(node1, node2) or graph.graph.has_edge(node2, node1):
                    connections += 1
                    
        connectivity_similarity = connections / (len(comm1_nodes) * len(comm2_nodes))
        
        # Combine similarities
        similarity = 0.7 * center_similarity + 0.3 * connectivity_similarity
        
        return similarity