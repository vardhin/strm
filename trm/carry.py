import torch
from dataclasses import dataclass

@dataclass
class SymbolicTRMCarry:
    """Carry state for TRM recursive reasoning."""
    z_H: torch.Tensor  # High-level reasoning state
    z_L: torch.Tensor  # Low-level reasoning state
    steps: torch.Tensor
    halted: torch.Tensor