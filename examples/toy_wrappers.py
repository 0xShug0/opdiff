# examples/toy_wrappers.py
import torch
import torch.nn as nn
from examples.toy_model import ToyConfig, ToyLN, ToyRMS


class ToyLogitsLN(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.model = ToyLN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ToyLogitsRMS(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.model = ToyRMS(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)