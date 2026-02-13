"""
Torch runtime helpers for determinism, device selection, and tensor I/O handling.

This module provides small utilities commonly used by runners/backends:

- `with_torch_seed(seed)`: context manager that forks RNG state and sets a local seed
  for deterministic sampling without affecting global RNG outside the context.
- `pick_device(device)`: resolve a user-facing device string ("cpu", "cuda", "mps", "gpu")
  into a `torch.device`, with clear errors when the requested backend is unavailable.
- Convenience helpers to cast/move tensor inputs and normalize outputs:
  - `to_fp16`: cast floating tensors to fp16
  - `move_tensor_args_to` / `move_tensor_kwargs_to`: move tensor inputs to a device
  - `to_cpu_out`: detach and move outputs to CPU recursively (tensors, lists, dicts)
"""

import contextlib
from typing import Any, Dict
import torch
        
@contextlib.contextmanager
def with_torch_seed(seed: int):
    """Temporarily set Torch RNG seed in an isolated RNG fork.

    Uses `torch.random.fork_rng` so RNG state changes are scoped to this context
    and do not leak to callers. Intended for deterministic sampling or test runs.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield

def pick_device(device: str) -> torch.device:
    """Resolve a device selector string into a concrete `torch.device`.

    Supported values:
    - "cpu": always available
    - "cuda": requires CUDA availability
    - "mps": requires Apple MPS availability
    - "gpu": prefers CUDA, then MPS

    Raises:
        TypeError: if `device` is not a string
        RuntimeError: if a requested accelerator is unavailable
        ValueError: if the device string is not recognized
    """
    if not isinstance(device, str):
        raise TypeError("device must be a string")

    d = device.lower()

    if d == "cpu":
        return torch.device("cpu")

    if d == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device 'cuda' but CUDA is not available")

    if d == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device 'mps' but MPS is not available")

    if d == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device 'gpu' but neither CUDA nor MPS is available")

    raise ValueError(
        f"Unknown device '{device}'. "
        "Supported values: 'cpu', 'cuda', 'mps', 'gpu'"
    )


def to_fp16(x):
    """Cast a floating-point tensor to fp16; leave non-floating values unchanged.

    Useful for quick mixed-precision experiments where only floating tensors
    should be downcast.
    """
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(torch.float16)
    return x


def move_tensor_args_to(dev: torch.device, xs):
    """Move positional tensor arguments to `dev`, leaving non-tensors untouched."""
    return [x.to(dev) if torch.is_tensor(x) else x for x in xs]


def move_tensor_kwargs_to(dev: torch.device, kw: Dict[str, Any]) -> Dict[str, Any]:
    """Move tensor values in a kwargs dict to `dev`, leaving non-tensors untouched."""
    out = {}
    for k, v in (kw or {}).items():
        out[k] = v.to(dev) if torch.is_tensor(v) else v
    return out


def to_cpu_out(y):
    """Detach tensors and move outputs to CPU recursively.

    Handles nested outputs commonly returned by ops/backends:
    - Tensor -> detached CPU tensor
    - list/tuple -> same container type with elements normalized
    - dict -> dict with values normalized
    - other types -> returned as-is
    """
    if torch.is_tensor(y):
        return y.detach().cpu()
    if isinstance(y, (tuple, list)):
        return type(y)(to_cpu_out(i) for i in y)
    if isinstance(y, dict):
        return {k: to_cpu_out(v) for k, v in y.items()}
    return y





