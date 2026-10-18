import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import sentence_transformers
except Exception:
    pass

from .embeddings import SemanticEncoder, get_encoder
from .config import FrameworkConfig, AgentConfig, CorticalSynthesizerConfig, RSALensConfig, JacobianLensConfig
from .api_queue import APIKeyQueue, ChatOpenAI
from .domain_registry import (
    DomainSpec,
    default_roster,
    load_roster,
    propose_roster,
    create_agents,
    domain_prior_distance,
)
from .parameter_registry import DomainParameterRegistry, AgentParameterRegistry, SCIENTIFIC_DOMAINS_11
from .rsa_engine import DefaultModeRSAEngine, ExternalRSALens
from .jacobian_lens import SalienceJacobianNetwork, ExternalJacobianLens
from .cortical_synthesizer import CentralExecutiveNetwork, CorticalSynthesizer
from .state import AgenticCortexState
from .framework import JacobianRSACortexFramework
from .dream_rsi import DreamRSILoop, ToolRegistry, synthesize_and_verify_tool

__all__ = [
    "FrameworkConfig",
    "AgentConfig",
    "CorticalSynthesizerConfig",
    "RSALensConfig",
    "JacobianLensConfig",
    "APIKeyQueue",
    "ChatOpenAI",
    "SemanticEncoder",
    "get_encoder",
    "DomainSpec",
    "default_roster",
    "load_roster",
    "propose_roster",
    "create_agents",
    "domain_prior_distance",
    "DomainParameterRegistry",
    "AgentParameterRegistry",
    "SCIENTIFIC_DOMAINS_11",
    "DefaultModeRSAEngine",
    "ExternalRSALens",
    "SalienceJacobianNetwork",
    "ExternalJacobianLens",
    "CentralExecutiveNetwork",
    "CorticalSynthesizer",
    "AgenticCortexState",
    "JacobianRSACortexFramework",
    "DreamRSILoop",
    "ToolRegistry",
    "synthesize_and_verify_tool",
]
