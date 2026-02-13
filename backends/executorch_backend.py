"""ExecuTorch backend implementation.

This backend exports a single op/module to an ExecuTorch program and runs it via
the ExecuTorch runtime.

Key behaviors:
- Export uses `torch.export.export`, optionally followed by decompositions.
- Lowering/delegation is selected by `ExecuTorchOptions.target` (cpu/cuda/mps/coreml).
- Complex tensors are packed as real-valued tensors with a trailing size-2
  dimension (..., 2) to keep runtime I/O strictly real-valued.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Any, List

import torch
import torch.nn as nn

# Hard-fail imports (per your requirement)
import executorch
from executorch.runtime import Runtime, Verification

# ExecuTorch 1.1+ export pipeline
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig

from backends.backend_base import Backend
from backends.backend_runners import BackendModuleRunner, BackendOpRunner
from backends.backend_utils import move_tensor_args_to, move_tensor_kwargs_to, to_cpu_out, with_torch_seed


@dataclass(frozen=True)
class TorchExportOptions:
    """Controls the torch.export path used prior to ExecuTorch lowering."""
    export_strict: bool = True
    export_dynamic_shapes: Any = None
    export_fp16: bool = False
    use_default_decompositions: bool = True


@dataclass(frozen=True)
class ExecuTorchOptions:
    """Controls lowering target, verification, and runtime execution behavior."""
     
    # Delegation/lowering target used by `to_edge_transform_and_lower`.
    # Supported: "cpu" | "cuda" | "mps" | "coreml"
    target: str = "cpu"
    
    # CoreML delegate setting (only used when target="coreml").
    coreml_compute_precision: str = "fp32"

    # Method name to load from the ExecuTorch program.
    method_name: str = "forward"
    
    # Runtime verification level applied when loading the program.
    verification: Verification = Verification.Minimal
    extract_delegate_segments: bool = True


class ExecuTorchBackend(Backend):
    """Backend that exports to ExecuTorch and runs with ExecuTorch runtime."""
    
    name = "executorch"

    def __init__(self, *, export_opts: TorchExportOptions, et_opts: ExecuTorchOptions, seed: int = 0):
        super().__init__(seed=seed)
        self.export_opts = export_opts
        self.et_opts = et_opts

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
        if self.export_opts.export_dynamic_shapes is not None:
            kwargs["dynamic_shapes"] = self.export_opts.export_dynamic_shapes
        return torch.export.export(model, args=inputs, **kwargs)

    def _get_partitioners_and_compile_config(self):
        """Select partitioners and compile config for the configured lowering target."""
        t = (self.et_opts.target or "cpu").lower()

        compile_config = None

        if t == "cpu":
            # XNNPACK delegate
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            return [XnnpackPartitioner()], compile_config

        if t == "cuda":
            # CUDA backend delegate
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
            from executorch.exir import EdgeCompileConfig

            compile_config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
            spec = CudaBackend.generate_method_name_compile_spec("opdiff")
            return [CudaPartitioner([spec])], compile_config

        if t == "coreml":
            # Apple CoreML delegate
            import coremltools as ct
            from executorch.backends.apple.coreml.compiler import CoreMLBackend
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
            prec = (self.et_opts.coreml_compute_precision or "fp32").lower()
            if prec == "fp16":
                compute_precision = ct.precision.FLOAT16
            elif prec == "fp32":
                compute_precision = ct.precision.FLOAT32
            compile_specs = CoreMLBackend.generate_compile_specs(
                compute_precision=compute_precision,
            )
            return [CoreMLPartitioner(compile_specs=compile_specs)], compile_config

        if t == "mps":
            # Apple MPS delegate
            from executorch.backends.apple.mps.partition import MPSPartitioner
            from executorch.exir.backend.backend_details import CompileSpec

            use_fp32 = not bool(self.export_opts.export_fp16)
            use_fp32_bytes = bytes([True]) if use_fp32 else bytes([False])
            return [MPSPartitioner([CompileSpec("use_fp32", use_fp32_bytes)])], compile_config

        raise ValueError(
            f"Unknown ExecuTorch target={self.et_opts.target!r} "
            f"(expected cpu/cuda/mps/coreml)"
        )

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
        device: str = "cpu",
        cast_input0_to_complex: bool = False,
    ):
        """Export an op/module to a .pte program and return a loaded runtime method."""
        dev = torch.device(device)

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

        # Match CoreMLBackend behavior: fp16 affects model dtype only.
        if self.export_opts.export_fp16:
            model = model.to(dtype=torch.float16)

        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".pte", prefix="opdiff_")
            os.close(fd)

        self._export(model, inputs, out_path)
        return self.load(out_path)
        
    def _export(self, model, inputs, out_path):
        """Lower an exported program to ExecuTorch and write it to `out_path` (timed)."""
        with torch.no_grad():
            exported = self._torch_export_call(model, inputs)
            if self.export_opts.use_default_decompositions:
                exported = exported.run_decompositions()
            else:
                exported = exported.run_decompositions({})

        partitioners, compile_config = self._get_partitioners_and_compile_config()

        edge_pm = to_edge_transform_and_lower(
            exported,
            partitioner=partitioners,
            compile_config=compile_config,
        )

        backend_config = ExecutorchBackendConfig(
            extract_delegate_segments=bool(self.et_opts.extract_delegate_segments)
        )
        et_pm = edge_pm.to_executorch(config=backend_config)
        with open(out_path, "wb") as f:
            et_pm.write_to_file(f)
            
    
    def load(self, path: str):
        """Load a .pte file and return a callable method object."""
        et_runtime = Runtime.get()
        program = et_runtime.load_program(
            Path(path),
            verification=self.et_opts.verification,
        )
        ptemodel = program.load_method(self.et_opts.method_name)
        return ptemodel

    def predict(self, model, tensor_args: Sequence[torch.Tensor]):
        """Run inference and normalize outputs to CPU tensors/structures."""
        inputs = self.pack_inputs(tensor_args)
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
            outs = model.execute(x)
        return outs