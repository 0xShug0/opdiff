"""ONNX Runtime backend implementation.

This backend exports a single op/module to ONNX via `torch.onnx.export`, then runs
inference with ONNX Runtime.

Notes:
- Complex tensors are represented as real-valued tensors with a trailing size-2
  dimension (..., 2) to keep ONNX I/O strictly real-valued.
- The backend returns raw ORT outputs (NumPy arrays). Higher-level runners may
  convert or compare these outputs as needed.
"""

from dataclasses import dataclass, field
import os
import tempfile
from typing import Any, Dict, List, Optional
import onnxruntime as ort
import torch
import numpy as np
from backends.backend_base import Backend
from backends.backend_runners import BackendOpRunner, BackendModuleRunner
from backends.backend_utils import move_tensor_args_to, move_tensor_kwargs_to, pick_device, with_torch_seed


@dataclass(frozen=True)
class ONNXOptions:
    """Controls ONNX export and ONNX Runtime session configuration."""
    opset: int = 18
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    export_fp16: bool = False
   
    # Export behavior
    dynamo: bool = True
    do_constant_folding: bool = True
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None  # or {"in0": {0:"B"}, "out0":{0:"B"}}
    external_data: bool = False
    
     # Kept for compatibility with existing configs (currently not used here).
    fallback_cpu: bool = True


class ONNXBackend(Backend):
    """Backend that exports to ONNX and runs inference with ONNX Runtime."""
    
    name = "onnx"

    def __init__(self, *, onnx_opts: ONNXOptions, seed: int = 0):
        super().__init__(seed=seed)
        self.onnx_opts = onnx_opts

    def wrap_op(self, op_fn, *, backend_mode: bool, cast_input0_to_complex: bool = False, **flags):
        """Optionally cast the first input back to complex during backend execution."""
        if backend_mode and cast_input0_to_complex:
            def _wrapped(x, *rest, **kwargs):
                x = self._to_complex_input0(x)
                return op_fn(x, *rest, **kwargs)
            return _wrapped
        return op_fn


    @staticmethod
    def _to_numpy_feed(t: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to a NumPy array suitable for ORT feeds."""
        return t.detach().cpu().numpy()

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
        """Export an op/module to ONNX and return a loaded ORT InferenceSession."""
        dev = pick_device(device)

        const_args = const_args or []
        const_kwargs = const_kwargs or {}
        kw_tensor_keys = kw_tensor_keys or []
        arg_is_tensor = arg_is_tensor or ([True] * len(example_tensor_inputs) + [False] * len(const_args))

        tensor_inputs = self.pack_inputs(example_tensor_inputs)
        tensor_inputs = move_tensor_args_to(dev, list(tensor_inputs))
        const_args2 = move_tensor_args_to(dev, list(const_args))
        const_kwargs2 = move_tensor_kwargs_to(dev, dict(const_kwargs))

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
        if self.onnx_opts.export_fp16:
            model = model.to(dtype=torch.float16)
        
        inputs = tuple(tensor_inputs)
        input_names = [f"in{i}" for i in range(len(tensor_inputs))]

        # Run once to determine output arity for stable naming.
        with torch.no_grad():
            y_ex = model(*inputs)
            
        if isinstance(y_ex, (tuple, list)):
            output_names = [f"out{i}" for i in range(len(y_ex))]
        else:
            output_names = ["out0"]

        if not out_path:
            fd, out_path = tempfile.mkstemp(suffix=".onnx", prefix="opdiff_")
            os.close(fd)

        self._export(model, inputs, input_names, output_names, out_path)
        return self.load(out_path)
    
    def _export(self, model, inputs, input_names, output_names, out_path):
        """Export model (timed)"""   
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                out_path,
                export_params=True,
                opset_version=int(self.onnx_opts.opset),
                do_constant_folding=bool(self.onnx_opts.do_constant_folding),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.onnx_opts.dynamic_axes,
                dynamo=bool(self.onnx_opts.dynamo),
                external_data=False,
            )

    def load(self, path: str):
        """Create an ORT session from an ONNX model path."""
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(path, sess_options=sess_opts, providers=self.onnx_opts.providers)
        return sess

    def predict(self, sess, tensor_args):
        """Run inference in ORT and return outputs (NumPy arrays)."""
        packed = self.pack_inputs(tensor_args)
        inputs = sess.get_inputs()

        if len(inputs) == 0:
            outs = self._infer(sess, {})
            return outs[0] if len(outs) == 1 else tuple(outs)

        if len(inputs) != len(packed):
            raise RuntimeError(f"ORT input count mismatch: session expects {len(inputs)} inputs, got {len(packed)}")

        feeds = {}
        for i, inp in enumerate(inputs):
            t = packed[i]
            if not torch.is_tensor(t):
                raise TypeError(f"Expected Tensor input at position {i}, got {type(t)}")
            feeds[inp.name] = self._to_numpy_feed(t)

        outs = self._infer(sess, feeds)
        if not outs:
            raise RuntimeError("ORT produced no outputs")
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _infer(self, sess, x):
        """Internal inference entrypoint (timed)."""
        return sess.run(None, x)
    