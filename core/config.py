from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AgentConfig:
    name: str
    domain: str
    description: str
    input_dim: int = 128
    hidden_dim: int = 256
    adapter_dim: int = 128
    num_soft_tokens: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3

@dataclass
class CorticalSynthesizerConfig:
    latent_dim: int = 128
    num_attention_heads: int = 4
    feedforward_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    temperature: float = 1.0

@dataclass
class RSALensConfig:
    metric: str = "cosine"
    diversity_margin: float = 0.35
    target_alignment_weight: float = 0.5
    eps: float = 1e-8

@dataclass
class JacobianLensConfig:
    compute_full_jacobian: bool = True
    spectral_norm_regularization: bool = True
    frobenius_weight: float = 0.01
    perturbation_epsilon: float = 1e-3

@dataclass
class FrameworkConfig:
    latent_dim: int = 128
    synthesizer: CorticalSynthesizerConfig = field(default_factory=CorticalSynthesizerConfig)
    rsa: RSALensConfig = field(default_factory=RSALensConfig)
    jacobian: JacobianLensConfig = field(default_factory=JacobianLensConfig)
    
    task_loss_weight: float = 1.0
    rsa_loss_weight: float = 0.25
    jacobian_loss_weight: float = 0.05
    confidence_loss_weight: float = 0.1
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    
    device: str = "cpu"
