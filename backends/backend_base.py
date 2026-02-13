"""Backend interface and shared utilities.

This module defines a minimal backend API used by runner implementations
(e.g., Torch, ONNX, CoreML, ExecuTorch). It also provides common helpers for:

- Timing critical backend operations (export/load/infer)
- Packing/unpacking complex tensors into real-valued representations
- Sanitizing backend outputs into torch-friendly structures
"""

import functools
import numbers
import time
import torch
from dataclasses import is_dataclass, fields

class Backend:
    """Base backend interface.

    Subclasses are expected to implement the core methods:
    - export_op(...)
    - predict(...)
    - load(...)
    - _infer(...)
    - _export(...)

    The base class automatically wraps selected methods to record runtime stats.
    """
    name: str = "noop"

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        # Stores the most recent timing for each key, in milliseconds.
        self.runtime_stats: dict[str, float] = {}
        self._wrap_timer("_infer", "infer")
        self._wrap_timer("_export", "export")
        self._wrap_timer("load", "load")

    def set_seed(self, seed: int) -> None:
        """Set the backend seed used by implementations that support determinism."""
        self.seed = int(seed)
    
    def _wrap_timer(self, method_name: str, key: str) -> None:
        """Wrap a method to record execution time in `runtime_stats` (milliseconds)."""
        fn = getattr(self, method_name, None)
        if fn is None or not callable(fn):
            return
        if getattr(fn, "__timed__", False):
            return

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.synchronize()
            self.runtime_stats[key] = (time.perf_counter() - t0) * 1000.0
            return out

        wrapped.__timed__ = True
        setattr(self, method_name, wrapped)

    def get_runtime_stats(self):
        """Return a copy of the last recorded timing values (milliseconds)."""
        return dict(self.runtime_stats)

    def get_runtime_stats_by_key(self, key: str, default: float | None = None):
        """Return the last recorded timing value for `key` (milliseconds)."""
        return self.runtime_stats.get(key, default)


    def wrap_op(self, op_fn, *, backend_mode: bool, **flags):
        """Optionally wrap an op callable for backend-specific behavior.

        Backends can override this to:
        - Insert tracing/compilation logic
        - Swap implementations based on flags
        - Apply autocast/device placement policies
        """
        return op_fn

    def synchronize(self) -> None:
        """Block until backend work completes (override for async runtimes)."""
        return 
    
    def descriptor(self) -> str:
        """Build a stable descriptor string for this backend and its option objects."""
        parts = [self.name]

        # include all option objects stored on self.* that end with "_opts"
        for attr, opts in sorted(self.__dict__.items()):
            if not attr.endswith("_opts"):
                continue

            if is_dataclass(opts):
                for f in fields(opts):
                    v = getattr(opts, f.name)
                    v = getattr(v, "name", v)  # nicer for enums/targets
                    parts.append(f"{f.name}={v}")
            else:
                parts.append(f"{attr}={opts}")

        return "; ".join(parts)
    
    def backend_name(self) -> str:
        """Stable backend identifier (e.g., 'torch', 'onnx', 'coreml')."""
        return self.name
    
    @staticmethod
    def _to_complex_input0(x):
        """Convert input 0 to a complex tensor when requested.

        Behavior-preserving conversion rules:
        - If `x` is a real tensor packed as (..., 2), interpret it as complex(re, im).
        - Else if `x` is a real floating tensor, convert to complex(x, 0).
        - Else if `x` is an int/bool tensor, cast to float32 then convert to complex(x, 0).
        - Otherwise return `x` unchanged.
        """
        if torch.is_tensor(x) and (not torch.is_complex(x)):
            if x.ndim >= 1 and x.shape[-1] == 2:
                return torch.complex(x[..., 0], x[..., 1])

            if x.dtype in (torch.float16, torch.float32, torch.float64):
                return torch.complex(x, torch.zeros_like(x))

            if x.dtype in (torch.int32, torch.int64, torch.bool):
                x2 = x.to(torch.float32)
                return torch.complex(x2, torch.zeros_like(x2))

        return x
    
    def sanitize_output(self, x):
        """Normalize backend outputs into torch-friendly values.

        - Complex tensors are returned as real-packed (..., 2) tensors.
        - Python scalars are converted to torch tensors with stable dtypes.
        - Lists/tuples/dicts are processed recursively.
        """
        if isinstance(x, torch.Tensor):
            if torch.is_complex(x):
                if x.is_conj():
                    x = x.conj()
                return torch.view_as_real(x)
            return x

        if isinstance(x, bool):
            return torch.tensor(x, dtype=torch.bool)
        if isinstance(x, numbers.Integral):
            return torch.tensor(int(x), dtype=torch.int64)
        if isinstance(x, numbers.Real):
            return torch.tensor(float(x), dtype=torch.float32)

        if isinstance(x, (tuple, list)):
            t = [self.sanitize_output(i) for i in x]
            return type(x)(t)
        if isinstance(x, dict):
            return {k: self.sanitize_output(v) for k, v in x.items()}

        return x

    def pack_inputs(self, tensor_args):
        """Pack complex tensor inputs into real-valued representations."""
        packed = []
        for t in tensor_args:
            if torch.is_tensor(t) and torch.is_complex(t):
                if t.is_conj():
                    t = t.conj()
                packed.append(torch.view_as_real(t))
            else:
                packed.append(t)
        return packed

    def export_op(
        self,
        *,
        op: str,
        example_tensor_inputs,
        const_args,
        arg_is_tensor,
        const_kwargs=None,
        kw_tensor_keys=None,
        out_path=None,
        device: str = "cpu",
        cast_input0_to_complex: bool = False,
    ):
        """Export/compile a single op for this backend."""
        raise NotImplementedError

    def predict(self, model, tensor_args):
        """Run inference for a previously exported/loaded model."""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load an exported model artifact from disk."""
        raise NotImplementedError
    
    def _infer(self, *args, **kwargs):
        """Backend-private infer entrypoint (timed if present)."""
        raise NotImplementedError
    
    def _export(self, *args, **kwargs):
        """Backend-private export entrypoint (timed if present)."""
        raise NotImplementedError


class NoopBackend(Backend):
    """A backend that performs no backend-specific behavior."""
    name = "noop"
