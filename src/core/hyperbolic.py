import torch
import math

class HyperbolicSpace:
    """Enhanced hyperbolic geometry implementation using the Lorentz model"""
    def __init__(self, dim: int = 3, curvature: float = -1.0):
        self.dim = dim
        self.curvature = curvature
        self.eps = 1e-15  # Numerical stability
        
    def minkowski_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski inner product with better numerical stability"""
        if x.dim() < y.dim():
            x = x.unsqueeze(0)
        if y.dim() < x.dim():
            y = y.unsqueeze(0)
            
        return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the hyperbolic manifold"""
        norm = self.minkowski_norm(x)
        denom = torch.sqrt(torch.abs(norm) + self.eps)
        return x / denom.unsqueeze(-1)
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski norm with sign"""
        return self.minkowski_dot(x, x)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance with numerical safeguards"""
        dot_product = -self.minkowski_dot(x, y)
        # Clip for numerical stability
        dot_product = torch.clamp(dot_product, min=1.0 + self.eps)
        return torch.acosh(dot_product) / math.sqrt(-self.curvature)
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Enhanced exponential map with better handling of edge cases and shapes
        
        Args:
            x: Base point tensor of arbitrary shape (..., dim)
            v: Tangent vector tensor of arbitrary shape (..., dim)
        Returns:
            Tensor on the manifold with same shape as input
        """
        # Ensure inputs have same shape
        if x.shape != v.shape:
            raise ValueError(
                f"Shape mismatch in exp_map: x shape {x.shape}, v shape {v.shape}"
            )
        
        # Save original shape for later reshape
        original_shape = x.shape
        
        # Reshape tensors to 2D for easier handling while preserving batch dimensions
        flat_x = x.view(-1, original_shape[-1])
        flat_v = v.view(-1, original_shape[-1])
        
        # Compute norms of tangent vectors
        v_norm = torch.norm(flat_v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=self.eps)  # Prevent division by zero
        
        sqrt_c = math.sqrt(-self.curvature)
        scaled_norm = sqrt_c * v_norm
        
        # Initialize result with x for numerical stability
        result = flat_x.clone()
        
        # Only compute for non-zero vectors
        nonzero_mask = v_norm.squeeze(-1) > self.eps
        if nonzero_mask.any():
            # Compute only for non-zero vectors
            v_normalized = flat_v[nonzero_mask] / v_norm[nonzero_mask]
            coeff = torch.cosh(scaled_norm[nonzero_mask])
            sinh_term = torch.sinh(scaled_norm[nonzero_mask])
            
            result[nonzero_mask] = (coeff * flat_x[nonzero_mask] + 
                                sinh_term * v_normalized)
        
        # Project back to the manifold
        result = self.project(result)
        
        # Restore original shape
        return result.view(original_shape)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Enhanced logarithmic map with better numerical stability"""
        dist = self.distance(x, y)
        # Handle small distances differently
        small_dist_mask = dist < self.eps
        
        result = torch.zeros_like(y)
        if small_dist_mask.any():
            result[small_dist_mask] = y[small_dist_mask] - x[small_dist_mask]
            
        # Regular computation for normal cases
        normal_mask = ~small_dist_mask
        if normal_mask.any():
            sqrt_c = math.sqrt(-self.curvature)
            alpha = (dist[normal_mask] * sqrt_c / 
                    torch.sinh(dist[normal_mask] * sqrt_c))
            result[normal_mask] = alpha.unsqueeze(-1) * (y[normal_mask] + 
                self.minkowski_dot(x[normal_mask], y[normal_mask]).unsqueeze(-1) * x[normal_mask])
            
        return result
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport with improved numerical handling"""
        dot_xy = self.minkowski_dot(x, y)
        dot_xv = self.minkowski_dot(x, v)
        
        # Handle near-identity cases
        near_identity = torch.abs(dot_xy - 1) < self.eps
        result = torch.zeros_like(v)
        
        if near_identity.any():
            result[near_identity] = v[near_identity]
            
        # Regular computation for normal cases
        normal_mask = ~near_identity
        if normal_mask.any():
            denom = 1 - dot_xy[normal_mask]
            coef = dot_xv[normal_mask] / denom
            result[normal_mask] = (v[normal_mask] + 
                                 coef.unsqueeze(-1) * (x[normal_mask] + y[normal_mask]))
            
        return self.project(result)