"""MLX backend implementation (work in progress).

This backend is under active development and may be incomplete or unstable.
Expect missing operator coverage and API changes as the implementation evolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from backends.backend_base import Backend
from backends.backend_utils import to_cpu_out, with_torch_seed

import mlx.core as mx


@dataclass(frozen=True)
class MlxRunOptions:
    # MLX uses Apple Silicon GPU by default; keep this for symmetry / future knobs.
    force_eval: bool = True


def _to_mx(x: Any) -> Any:
    if torch.is_tensor(x):
        return mx.array(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return mx.array(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_mx(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_mx(v) for k, v in x.items()}
    return x


def _mx_to_torch_cpu(x: Any) -> Any:
    if isinstance(x, mx.array):
        arr = np.array(x)
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

    spec = path[len("file:") :]
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

        # Support local imports relative to the file directory (same behavior as torch resolver).
        file_dir = os.path.dirname(file_abspath)
        restore_sys_path = False
        if file_dir and file_dir not in sys.path:
            sys.path.insert(0, file_dir)
            restore_sys_path = True
        try:
            module_spec.loader.exec_module(mod)
        finally:
            if restore_sys_path:
                if sys.path and sys.path[0] == file_dir:
                    sys.path.pop(0)
                else:
                    try:
                        sys.path.remove(file_dir)
                    except ValueError:
                        pass

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
    """MLX backend (module-only, file:...::Symbol supported)."""

    name = "mlx"

    def __init__(self, *, run_opts: MlxRunOptions | None = None, seed: int = 0):
        super().__init__(seed=seed)
        self.run_opts = run_opts or MlxRunOptions()

    def synchronize(self) -> None:
        return

    def export_op(self, op):
        """
        Build a runnable object from a module spec (ModuleNode-like).

        For type='module', treat the resolved symbol as a constructor/factory:
          model = symbol(*op.args, **op.kwargs)
        """
        if isinstance(op, str):
            raise NotImplementedError(
                "MLX backend supports only module specs (type='module'), not string ops."
            )

        ctor = _resolve_mlx_path(op.path)
        ctor_args = list(op.args or [])
        ctor_kwargs = dict(op.kwargs or {})

        with with_torch_seed(self.seed):
            model = ctor(*ctor_args, **ctor_kwargs)

        return model

    def predict(self, model, inputs, *, kwargs=None):
        kwargs = kwargs or {}
        mx_inputs = _to_mx(inputs)
        mx_kwargs = _to_mx(kwargs)

        out = self._infer(model, mx_inputs, mx_kwargs)

        out_torch = _mx_to_torch_cpu(out)
        return to_cpu_out(out_torch)

    def _infer(self, model, mx_inputs, mx_kwargs):
        with with_torch_seed(self.seed):
            out = model(*mx_inputs, **mx_kwargs)
        mx.eval(out)
        return out
