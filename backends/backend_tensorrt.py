"""TensorRT backend implementation.

This backend exports a single op/module through torch.export and compiles it with
Torch-TensorRT using the Dynamo/AOT path. The compiled program is serialized to
disk and can be loaded back for inference.

"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

# Hard-fail imports
import torch_tensorrt

from backends.backend_base import Backend
from backends.backend_runners import BackendModuleRunner, BackendOpRunner
from backends.backend_utils import (
    move_tensor_args_to,
    move_tensor_kwargs_to,
    pick_device,
    to_cpu_out,
    with_torch_seed,
)


@dataclass(frozen=True)
class TorchExportOptions:
    """Controls the torch.export path used prior to TensorRT compilation."""
    export_strict: bool = True
    export_fp16: bool = False
    use_default_decompositions: bool = True


@dataclass(frozen=True)
class TensorRTOptions:
    """Controls TensorRT compilation and runtime execution behavior."""
    device: str = "cuda"
    use_explicit_typing: bool = True
    min_block_size: int = 5
    cache_built_engines: bool = False
    reuse_cached_engines: bool = False
    hardware_compatible: bool = False
    version_compatible: bool = False
    optimization_level: Optional[int] = None
    max_aux_streams: Optional[int] = None
    


class TensorRTBackend(Backend):
    """Backend that exports to Torch-TensorRT and runs with TensorRT on CUDA."""

    name = "tensorrt"

    def __init__(self, *, export_opts: TorchExportOptions, trt_opts: TensorRTOptions, seed: int = 0):
        super().__init__(seed=seed)
        self.export_opts = export_opts
        self.trt_opts = trt_opts

    def synchronize(self) -> None:
        """Synchronize CUDA execution for accurate timing."""
        dev = pick_device(self.trt_opts.device)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def wrap_op(self, op_fn, *, backend_mode: bool, cast_input0_to_complex: bool = False, **flags):
        """Optionally cast the first input back to complex during backend execution."""
        if backend_mode and cast_input0_to_complex:
            def _wrapped(x, *rest, **kwargs):
                x = self._to_complex_input0(x)
                return op_fn(x, *rest, **kwargs)
            return _wrapped
        return op_fn

    def _torch_export_call(self, model: nn.Module, inputs: tuple):
        """Export a model with torch.export.export using configured options."""
        kwargs = {}
        kwargs["strict"] = bool(self.export_opts.export_strict)
        return torch.export.export(model, args=inputs, **kwargs)

    def export_op(
        self,
        *,
        op: Any,
        example_tensor_inputs: Sequence[torch.Tensor],
        const_args=None,
        arg_is_tensor=None,
        const_kwargs=None,
        kw_tensor_keys=None,
        out_path: Optional[str] = None,
        device: str = "cuda",
        cast_input0_to_complex: bool = False,
    ):
        """Export an op/module to a TensorRT artifact and return a loaded runnable model. Always use cuda"""
        _ = device
        dev = pick_device(self.trt_opts.device)
        if dev.type != "cuda":
            raise ValueError(f"TensorRTBackend requires a CUDA device, got {device!r}")

        const_args = const_args or []
        const_kwargs = const_kwargs or {}
        kw_tensor_keys = kw_tensor_keys or []
        arg_is_tensor = arg_is_tensor or ([True] * len(example_tensor_inputs) + [False] * len(const_args))

        tensor_inputs = self.pack_inputs(example_tensor_inputs)
        tensor_inputs = move_tensor_args_to(dev, tensor_inputs)
        const_args2 = move_tensor_args_to(dev, const_args)
        const_kwargs2 = move_tensor_kwargs_to(dev, const_kwargs)
        inputs = tuple(tensor_inputs)

        with with_torch_seed(self.seed):
            if isinstance(op, str):
                model = BackendOpRunner(
                    op,
                    const_args=const_args2,
                    arg_is_tensor=arg_is_tensor,
                    const_kwargs=const_kwargs2,
                    kw_tensor_keys=kw_tensor_keys,
                    cast_input0_to_complex=cast_input0_to_complex,
                    runner=self,
                )
            else:
                model = BackendModuleRunner(
                    op,
                    const_args=const_args2,
                    arg_is_tensor=arg_is_tensor,
                    const_kwargs=const_kwargs2,
                    kw_tensor_keys=kw_tensor_keys,
                    runner=self,
                )

        model = model.eval().to(dev)

        if self.export_opts.export_fp16:
            model = model.to(dtype=torch.float16)

        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".pt2", prefix="opdiff_trt_")
            os.close(fd)

        self._export(model, inputs, out_path)
        return self.load(out_path)

    def _export(self, model, inputs, out_path):
        """Compile an exported program with Torch-TensorRT and write it to `out_path`."""
        with torch.no_grad():
            exported = self._torch_export_call(model, inputs)
            if self.export_opts.use_default_decompositions:
                exported = exported.run_decompositions()
            else:
                exported = exported.run_decompositions({})

        compile_kwargs = {
            "inputs": list(inputs),
            "use_explicit_typing": bool(self.trt_opts.use_explicit_typing),
            "min_block_size": int(self.trt_opts.min_block_size),
            "cache_built_engines": bool(self.trt_opts.cache_built_engines),
            "reuse_cached_engines": bool(self.trt_opts.reuse_cached_engines),
            "hardware_compatible": bool(self.trt_opts.hardware_compatible),
            "version_compatible": bool(self.trt_opts.version_compatible),
        }

        if self.trt_opts.optimization_level is not None:
            compile_kwargs["optimization_level"] = int(self.trt_opts.optimization_level)
        if self.trt_opts.max_aux_streams is not None:
            compile_kwargs["max_aux_streams"] = int(self.trt_opts.max_aux_streams)

        compiled = torch_tensorrt.dynamo.compile(exported, **compile_kwargs)

        torch_tensorrt.save(
            compiled,
            out_path,
            output_format="exported_program",
            arg_inputs=list(inputs),
        )

    def load(self, path: str):
        """Load a serialized Torch-TensorRT artifact."""
        return torch_tensorrt.load(path).module()

    def predict(self, model, tensor_args: Sequence[torch.Tensor]):
        """Run inference and normalize outputs to CPU tensors/structures."""
        dev = pick_device(self.trt_opts.device)
        if dev.type != "cuda":
            raise ValueError(f"TensorRTBackend requires a CUDA device, got {self.trt_opts.device!r}")

        inputs = self.pack_inputs(tensor_args)
        inputs = move_tensor_args_to(dev, inputs)

        outs = self._infer(model, inputs)
        outs = to_cpu_out(outs)

        if isinstance(outs, (list, tuple)):
            if len(outs) == 1:
                return outs[0]
            return tuple(outs)
        return outs

    def _infer(self, model, x):
        """Internal inference entrypoint (timed)."""
        with torch.no_grad():
            outs = model(*x)
        return outs