"""MLX backend implementation (work in progress).

This backend is under active development and may be incomplete or unstable.
Expect missing operator coverage and API changes as the implementation evolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from backends.backend_base import Backend
from backends.backend_utils import to_cpu_out, with_torch_seed

import mlx.core as mx


@dataclass(frozen=True)
class MlxRunOptions:
    # MLX is “device-less” in the same way; it uses Apple Silicon GPU by default.
    # Keep this for symmetry / future knobs.
    force_eval: bool = True
    
def _to_mx(x: Any) -> Any:
    if torch.is_tensor(x):
        # torch -> numpy -> mx
        return mx.array(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return mx.array(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_mx(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_mx(v) for k, v in x.items()}
    return x


def _mx_eval_tree(x: Any) -> None:
    if isinstance(x, mx.array):
        mx.eval(x)
        return
    if isinstance(x, (list, tuple)):
        for v in x:
            _mx_eval_tree(v)
        return
    if isinstance(x, dict):
        for v in x.values():
            _mx_eval_tree(v)
        return


def _mx_to_torch_cpu(x: Any) -> Any:
    # Convert mx arrays to torch tensors on CPU; keep structure.
    if isinstance(x, mx.array):
        arr = np.array(x)  # forces materialization if not already eval'd
        return torch.from_numpy(arr)
    if isinstance(x, (list, tuple)):
        return type(x)(_mx_to_torch_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _mx_to_torch_cpu(v) for k, v in x.items()}
    return x


def _load_file_symbol(path: str):
    # Expects: file:relative/or/abs.py::Symbol
    import importlib.util
    import os
    import sys

    if not path.startswith("file:"):
        raise ValueError(f"MLX backend only supports file:...::Symbol paths. Got: {path}")

    spec = path[len("file:"):]
    file_part, sep, sym = spec.partition("::")
    if not sep or not file_part or not sym:
        raise ValueError(f"Invalid file path spec: {path} (expected file:...py::Symbol)")

    file_abspath = os.path.abspath(file_part)
    mod_name = f"_opdiff_file_{abs(hash(file_abspath))}"

    mod = sys.modules.get(mod_name)
    if mod is None:
        module_spec = importlib.util.spec_from_file_location(mod_name, file_abspath)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load module from {file_abspath}")
        mod = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = mod
        module_spec.loader.exec_module(mod)

    obj = getattr(mod, sym, None)
    if obj is None:
        raise AttributeError(f"{path}: symbol '{sym}' not found in {file_abspath}")
    return obj

def _resolve_mlx_path(path: str):
    if path.startswith("file:"):
        return _load_file_symbol(path)

    if path.startswith("mlx."):
        obj = __import__("mlx")
        for p in path.split(".")[1:]:
            obj = getattr(obj, p)
        return obj

    raise ValueError(f"Unsupported MLX path: {path} (expected file:...::Symbol or mlx.* dotted path)")

class MlxBackend(Backend):
    """MLX backend (work in progress)."""
    
    name = "mlx"

    def __init__(self, *, run_opts: MlxRunOptions | None = None, seed: int = 0):
        super().__init__(seed=seed)
        self.run_opts = run_opts or MlxRunOptions()
        
        # cached state for infer()
        self._callable = None
        self._call_args = None
        self._call_kwargs = None
    
    def warmup(self, op, inputs, *, kwargs=None):
        kwargs = kwargs or {}

        # Convert inputs/kwargs to MLX tensors
        mx_inputs = _to_mx(inputs)
        mx_kwargs = _to_mx(kwargs)

        if isinstance(op, str):
            raise NotImplementedError(
                "MLX backend currently supports only module specs (ModuleNode-like), not string ops."
            )

        fn = _resolve_mlx_path(op.path)

        call_args = list(op.args or [])
        call_kwargs = dict(op.kwargs or {})

        with with_torch_seed(self.seed):
            if isinstance(fn, type):
                # class: op.args/op.kwargs are constructor args/kwargs
                mod = fn(*call_args, **call_kwargs)
                # cache callable + args for infer()
                self._callable = mod
                self._call_args = tuple(mx_inputs)
                self._call_kwargs = dict(mx_kwargs)

                out = self._infer()
                
            else:
                # function: op.args/op.kwargs are call-time args/kwargs (like TorchModuleRunner)
                merged_kwargs = dict(call_kwargs)
                merged_kwargs.update(mx_kwargs)

                # cache callable + args for infer()
                self._callable = fn
                self._call_args = tuple(mx_inputs) + tuple(call_args)
                self._call_kwargs = merged_kwargs

                out = self._infer()
      
        # Convert output to torch CPU tensors to fit your existing diff + reporting
        out_torch = _mx_to_torch_cpu(out)
        return to_cpu_out(out_torch)
    
    def _infer(self):
        with with_torch_seed(self.seed):
            out = self._callable(*self._call_args, **self._call_kwargs)

        # Force execution (MLX is lazy) - keep in infer so each timed call does real work
        mx.eval(out)
        return out