from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch

@dataclass
class ModelConfig:
    """Enhanced configuration for the model architecture"""
    dim: int = 768  # Embedding dimension
    max_vocab_size: int = 50000
    num_communities: int = 100
    tensor_bond_dim: int = 64
    fractal_code_dim: int = 128
    num_attention_heads: int = 8
    dropout: float = 0.1
    activation: str = 'gelu'
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    vocab_size: int = 50000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    edge_importance_threshold: float = 0.5
    input_dim: int = 3  # Number of input dimensions
    pattern_dim: int = 4  # Number of pattern dimensions
    attention_dim: int = 3  # Required attention dimensions
    pattern_top_k: int = 10  # Number of patterns to compare
    num_patterns: int = 1000  # Number of patterns to store in memory
    pattern_learning_rate: float = 0.1  # Learning rate for pattern updates
    pattern_size: int = 32  # Size of pattern representations
    max_batch_tokens: int = 12288  # Maximum number of tokens per batch

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    num_epochs: int = 10
    batch_size: int = 32
    accumulation_steps: int = 1
    logging_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 1000
    max_grad_norm: float = 1.0
    use_wandb: bool = True
    checkpoint_dir: str = "checkpoints"

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    vocab_size: int = 50000
    min_frequency: int = 2
    special_tokens: List[str] = None
    max_length: int = 512
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    backend: str = 'nccl'  # or 'gloo' for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    use_horovod: bool = False

@dataclass
class OptimizationConfig:
    """Configuration for training optimization"""
    lr_range: Tuple[float, float] = (1e-5, 1e-3)
    batch_size_range: Tuple[int, int] = (16, 128)
    weight_decay_range: Tuple[float, float] = (0.0, 0.1)
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    n_trials: int = 100
    epochs_per_trial: int = 10
    optimization_metric: str = 'validation_loss'
    storage_url: Optional[str] = None
    pruner_warmup_steps: int = 5


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive optimization"""
    lr_bounds: Tuple[float, float] = (1e-6, 1e-2)
    batch_size_bounds: Tuple[int, int] = (8, 256)
    momentum_bounds: Tuple[float, float] = (0.8, 0.999)
    window_size: int = 100
    adaptation_interval: int = 10
    target_loss_change: float = -0.01
    min_steps_per_adjust: int = 50


@dataclass
class ServingConfig:
    """Configuration for model serving"""
    model_name: str
    version: str
    batch_size: int = 32
    max_sequence_length: int = 512
    use_tensorrt: bool = False
    use_dynamic_batching: bool = True
    max_batch_delay: float = 0.1  # seconds
    cache_size: int = 1000
    timeout: float = 5.0  # seconds
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    project_name: str
    experiment_name: str
    tags: List[str] = field(default_factory=list)
    track_artifacts: bool = True
    log_interval: int = 100
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    visualize_results: bool = True

@dataclass
class PipelineConfig:
    """Configuration for pipeline management"""
    pipeline_name: str
    max_workers: int = 4
    timeout: float = 3600  # 1 hour
    retry_attempts: int = 3
    retry_delay: float = 60  # 1 minute
    log_level: str = "INFO"
    artifacts_dir: str = "artifacts"


@dataclass
class TestConfig:
    """Configuration for test framework"""
    test_data_path: str
    model_path: str
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 512])
    num_stress_iterations: int = 100
    performance_threshold: float = 0.1  # 100ms
    gpu_memory_threshold: float = 0.9  # 90% utilization
    num_workers: int = 4

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    model_variants: List[str]
    batch_sizes: List[int]
    sequence_lengths: List[int]
    num_iterations: int = 100
    warmup_iterations: int = 10
    metrics_output_path: str = "benchmark_results"
    compare_baseline: bool = True
    baseline_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PRValidationConfig:
    """Configuration for PR validation"""
    github_token: str
    repository: str
    required_checks: List[str]
    size_limits: Dict[str, int]
    notification_channels: Dict[str, str]
    auto_merge_labels: List[str]
    required_reviewers: int = 2