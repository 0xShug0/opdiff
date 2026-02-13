"""CoreML backend implementation.

This backend exports PyTorch ops/modules to CoreML (ML Program or NeuralNetwork)
using coremltools. It supports two export paths:

1) torch.export.export (+ optional decompositions) -> coremltools.convert
2) torch.jit.trace -> coremltools.convert (fallback)

Complex tensors are packed as real-valued (..., 2) tensors to keep CoreML inputs
and outputs strictly real-valued.
"""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from backends.backend_base import Backend
from backends.backend_runners import BackendModuleRunner, BackendOpRunner
from backends.backend_utils import move_tensor_args_to, move_tensor_kwargs_to, pick_device, with_torch_seed
import coremltools as ct

@dataclass(frozen=True)
class TorchExportOptions:
    """Controls the torch.export path used prior to CoreML conversion."""
    export_strict: bool = True
    export_dynamic_shapes: Any = None # torch.export dynamic_shapes mapping or None
    export_fp16: bool = False
    use_default_decompositions: bool = True


@dataclass(frozen=True)
class CoreMLOptions:
    """Controls coremltools conversion and runtime execution settings."""
    convert_to: str = "mlprogram"
    minimum_ios_target: Any = ct.target.iOS16
    compute_units: Any = ct.ComputeUnit.ALL
    compute_precision: Any = None  # None -> do not pass; preserves current behavior
    
    
class CoreMLBackend(Backend):
    """Backend that exports to and runs CoreML models via coremltools."""
    
    name = "coreml"

    def __init__(self, *, export_opts: TorchExportOptions, coreml_opts: CoreMLOptions, seed: int = 0):
        super().__init__(seed=seed)
        self.export_opts = export_opts
        self.coreml_opts = coreml_opts


    @staticmethod
    def _input_dtype_from_tensor(t: torch.Tensor):
        """Map a torch dtype to a CoreML input signature dtype.

        CoreML model signatures commonly expect int32 for integer/bool inputs.
        """
        if t.dtype in (torch.int64, torch.bool):
            return np.int32
        if t.dtype == torch.int32:
            return np.int32
        if t.dtype == torch.float16:
            return np.float16
        return np.float32


    @staticmethod
    def _coreml_feed_array(arr: np.ndarray):
        """Normalize numpy arrays to dtypes CoreML runtime reliably accepts."""
        if arr.dtype == np.bool_:
            return arr.astype(np.int32)
        if arr.dtype == np.int64:
            return arr.astype(np.int32)
        if arr.dtype == np.int32:
            return arr
        if arr.dtype == np.float16:
            return arr
        return arr.astype(np.float32)


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
        # strict
        kwargs["strict"] = bool(self.export_opts.export_strict)
        # dynamic_shapes
        if self.export_opts.export_dynamic_shapes is not None:
            kwargs["dynamic_shapes"] = self.export_opts.export_dynamic_shapes

        return torch.export.export(model, args=inputs, **kwargs)
        

    def _ct_convert(self, prog, *, ct_inputs):
        """Convert a torch program/module to a CoreML MLModel."""
        kwargs = dict(
            convert_to=self.coreml_opts.convert_to,
            inputs=ct_inputs,
            compute_units=self.coreml_opts.compute_units,
            minimum_deployment_target=self.coreml_opts.minimum_ios_target,
        )
        if self.coreml_opts.compute_precision is not None:
            kwargs["compute_precision"] = self.coreml_opts.compute_precision

        return ct.convert(prog, **kwargs)

    def export_op(
        self,
        *,
        op,
        example_tensor_inputs,
        const_args,
        arg_is_tensor,
        const_kwargs=None,
        kw_tensor_keys=None,
        out_path=None,
        device: str = "cpu",
        cast_input0_to_complex: bool = False,
    ):
        """Export an op/module to CoreML and optionally save it to disk."""
        dev = pick_device(device)
        const_args = const_args or []
        const_kwargs = const_kwargs or {}
        kw_tensor_keys = kw_tensor_keys or []
        arg_is_tensor = arg_is_tensor or ([True] * len(example_tensor_inputs) + [False] * len(const_args))

        packed_inputs = self.pack_inputs(example_tensor_inputs)
        packed_inputs = move_tensor_args_to(dev, packed_inputs)
        const_args2 = move_tensor_args_to(dev, const_args)
        const_kwargs2 = move_tensor_kwargs_to(dev, const_kwargs)

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
    
        inputs = tuple(packed_inputs)

        ct_inputs = [
            ct.TensorType(shape=tuple(i.shape), dtype=self._input_dtype_from_tensor(i))
            for i in packed_inputs
            if torch.is_tensor(i)
        ]
        
        mlmodel = self._export(model, inputs, ct_inputs)
        
        if out_path is not None:
            mlmodel.save(out_path)
        
        return mlmodel

    def _export(self, model, inputs, ct_inputs):
        """Try torch.export first, then fall back to torch.jit.trace. (timed)"""
        exported = None
        with torch.no_grad():
            try:
                exported = self._torch_export_call(model, inputs)
                if self.export_opts.use_default_decompositions:
                    exported = exported.run_decompositions()
                else:
                    exported = exported.run_decompositions({})
            except Exception:
                exported = None
    
        if exported is not None:
            try:
                return self._ct_convert(exported, ct_inputs=ct_inputs)
            except Exception:
                pass

        with torch.no_grad():
            traced = torch.jit.trace(model, inputs)
        return self._ct_convert(traced, ct_inputs=ct_inputs)

    def load(self, path: str):
        """Load a saved CoreML model."""
        return ct.models.MLModel(path, compute_units=self.coreml_opts.compute_units)
    
    def predict(self, mlmodel, tensor_args):
        """Run inference and return outputs ordered by the CoreML spec."""
        packed = self.pack_inputs(tensor_args)
        
        spec_inputs = list(mlmodel._spec.description.input)
        feeds = {}
        for i, spec_in in enumerate(spec_inputs):
            t = packed[i]
            arr = t.detach().cpu().numpy()
            feeds[spec_in.name] = self._coreml_feed_array(arr)

        outs = self._infer(mlmodel, feeds)
        if not isinstance(outs, dict):
            return outs
     
        # Preserve CoreML output ordering defined by the model spec.
        spec_outs = list(mlmodel._spec.description.output)
        ordered = []
        for o in spec_outs:
            if o.name not in outs:
                raise KeyError(
                    f"CoreML predict missing expected output '{o.name}'. "
                    f"Got keys: {list(outs.keys())}"
                )
            ordered.append(outs[o.name])

        # If single output, return the array directly; else return a tuple
        return ordered[0] if len(ordered) == 1 else tuple(ordered)

    def _infer(self, mlmodel, x):
        """Run CoreML forward pass; returns the raw CoreML output (timed)."""
        out = mlmodel.predict(x)
        return out
    
