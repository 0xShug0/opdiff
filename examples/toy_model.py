# examples/toy_model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    eps: float = 1e-5
    seed: int = 0


class RMSNorm(nn.Module):
    """
    RMSNorm: x * rsqrt(mean(x^2) + eps) * weight + bias
    Single-branch, export-friendly.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = x / rms
        return y * self.weight + self.bias


class ToyLN(nn.Module):
    """
    Baseline model: Linear -> GELU -> LayerNorm -> Linear
    """
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.ln = nn.LayerNorm(cfg.hidden_dim, eps=cfg.eps)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x


class ToyRMS(nn.Module):
    """
    "Optimized" / alternative: Linear -> GELU -> RMSNorm -> Linear
    Same structure, swapped norm layer => different outputs.
    """
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.rms = RMSNorm(cfg.hidden_dim, eps=cfg.eps)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.rms(x)
        x = self.fc2(x)
        return x

class UnaryOp(nn.Module):
    """
    mode: 'abs' | 'neg' | 'square'
    """
    def __init__(self, mode: str = "abs"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "abs":
            return x.abs()
        if self.mode == "neg":
            return -x
        if self.mode == "square":
            return x * x
        raise ValueError(f"Unknown mode={self.mode!r}")