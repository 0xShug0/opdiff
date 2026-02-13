"""Torch backend implementation.

This backend executes ops and modules directly with PyTorch on a selected device.
It provides optional autocast and fp16 execution and reports timing for inference
via the base Backend timer wrapper.
"""

import contextlib
from dataclasses import dataclass
import torch
from backends.backend_base import Backend
from backends.backend_runners import TorchModuleRunner, TorchOpRunner
from backends.backend_utils import move_tensor_args_to, move_tensor_kwargs_to, pick_device, to_cpu_out, with_torch_seed


@dataclass(frozen=True)
class TorchRunOptions:
    """Runtime options for TorchBackend."""
    device: str = "cpu"                     # cpu | mps
    autocast: bool = False                  # enable mixed precision
    matmul_precision: str = "highest"  # "highest" | "high" | "medium"
    run_fp16: bool = False


class TorchBackend(Backend):
    """Backend that runs models using eager PyTorch execution."""
    
    name = "torch"

    def __init__(self, *, run_opts: TorchRunOptions | None = None, seed: int = 0):
        super().__init__(seed=seed)
        self.run_opts = run_opts or TorchRunOptions()
        
         # Apply global matmul precision policy for float32 matmul operations
        torch.set_float32_matmul_precision(self.run_opts.matmul_precision)
        
    def synchronize(self) -> None:
        """Synchronize device execution for accurate timing."""
        dev = pick_device(self.run_opts.device)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()
    
    def export_op(self, op):
        """Build a runnable torch.nn.Module for an op or module."""
        opts = self.run_opts
        dev = pick_device(opts.device)
        with with_torch_seed(self.seed):
            model = TorchOpRunner(op) if isinstance(op, str) else TorchModuleRunner(op.path, op.args, op.kwargs)

        model = model.eval().to(dev)

        if opts.run_fp16:
            model = model.to(dtype=torch.float16)
            
        return model
    
    def predict(self, model, inputs, *, kwargs=None):
        """Run a model forward pass and return outputs on CPU."""
        opts = self.run_opts
        dev = pick_device(opts.device)
        model = model.to(dev)
        inputs = move_tensor_args_to(dev, inputs)
        kwargs = move_tensor_kwargs_to(dev, kwargs or {})
        
        # Autocast uses the device's type (cpu/cuda/mps where supported).
        autocast_ctx = (
            torch.autocast(device_type=dev.type)
            if (opts.autocast)
            else contextlib.nullcontext()
        )
        
        out = self._infer(model, inputs, kwargs, autocast_ctx)
        return to_cpu_out(out)
    
    def _infer(self, model, inputs, kwargs, autocast_ctx):
        """Internal inference entrypoint (timed by Backend)."""
        with torch.no_grad():
            with autocast_ctx:
                out = model(*inputs, **kwargs)

        return out
    