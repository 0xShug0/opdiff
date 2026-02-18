"""
Repro summary:

This script reproduces an ExecuTorch CoreML-delegate runtime crash (segmentation fault) during inference for
models that execute `aten::where` (single-input form) or `aten::nonzero_numpy`.

Observed behavior:
- Export and lowering complete successfully.
- The ExecuTorch runtime loads the program and method successfully.
- The process crashes during `method.execute(...)` (inference) with a segmentation fault (exit code -11).

Triggering models:
- `Where1Input`: `torch.ops.aten.where.default(x)` (single tensor input). The exported graph lowers to
  `nonzero -> slice -> squeeze` and returns per-dimension index vectors.
- `NonzeroNumpy`: `torch.ops.aten.nonzero_numpy.default(x)`. The exported graph lowers to the same
  `nonzero -> slice -> squeeze` pattern and returns per-dimension index vectors.

Root-cause characterization (from minimized variants in this script):
- Returning dynamic int64 tensors with rank >= 2 works (e.g., `i64[u0,2]` from `aten.nonzero`,
  and `i64[u0,1]` from slicing it).
- Producing a dynamic-length 1D int64 tensor `i64[u0]` via `aten.squeeze.dims` triggers the crash, even when
  returning a single output (not a tuple).
- Dynamic-length 1D float output `f32[u0]` does not crash, suggesting the issue is specific to dynamic 1D int
  outputs (or int64) under the CoreML delegate/runtime.

Expected behavior:
- ExecuTorch should either execute successfully and return the correct outputs, or raise a Python-visible error
  if unsupported by the CoreML delegate/runtime.
- It should not terminate the process with a segmentation fault.

Minimal trigger:
- Input is a float32 tensor of shape [2, 4].
- Lowering target is the ExecuTorch CoreML delegate.
"""



import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import coremltools as ct
from executorch.runtime import Runtime, Verification
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner


class Where1Input(nn.Module): # Crash
    def forward(self, x):
        return torch.ops.aten.where.default(x)


class NonzeroNumpy(nn.Module): # Crash
    def forward(self, x):
        return torch.ops.aten.nonzero_numpy.default(x)

class Nonzero(nn.Module): # Pass
    def forward(self, x):
        return torch.ops.aten.nonzero.default(x)

class NonzeroSplitNoSqueeze(nn.Module): # Pass
    def forward(self, x):
        nz = torch.ops.aten.nonzero.default(x)
        r = torch.ops.aten.slice.Tensor(nz, 1, 0, 1)  # i64[u0,1]
        return r
    
class NonzeroSplitReturnRow(nn.Module): # Crash
    def forward(self, x):
        nz = torch.ops.aten.nonzero.default(x)
        r = torch.ops.aten.slice.Tensor(nz, 1, 0, 1)
        r = torch.ops.aten.squeeze.dims(r, [1])
        return r

class IndexBoolUnsqueezeSqueeze(nn.Module): # Crash
    def forward(self, x):
        m = torch.ops.aten.gt.Scalar(x, 0.0)                  # b8[2,4]
        y = torch.ops.aten.index.Tensor(x, [m])               # f32[u0]
        yi = torch.ops.aten._to_copy.default(y, dtype=torch.int64)  # i64[u0]
        yi2 = torch.ops.aten.unsqueeze.default(yi, 0)         # i64[1,u0]
        return torch.ops.aten.squeeze.dims(yi2, [0])          # i64[u0]

class IndexBoolUnsqueezeSqueezeF32(nn.Module): # Pass
    def forward(self, x):
        m = torch.ops.aten.gt.Scalar(x, 0.0)
        y = torch.ops.aten.index.Tensor(x, [m])          # f32[u0]
        y2 = torch.ops.aten.unsqueeze.default(y, 0)      # f32[1,u0]
        return torch.ops.aten.squeeze.dims(y2, [0])      # f32[u0]


    
def run_case(name, model, inputs):
    print(f"=== {name} ===")

    exported = torch.export.export(model, args=inputs, strict=True)
    print("1. export: torch.export.export OK")

    exported = exported.run_decompositions()
    print("2. export: graph")
    exported.graph_module.print_readable()

    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT32,
    )

    edge_pm = to_edge_transform_and_lower(
        exported,
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
        compile_config=None,
    )
    et_pm = edge_pm.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    print("3. lower: to_executorch OK")

    fd, p = tempfile.mkstemp(suffix=".pte", prefix="where_")
    os.close(fd)
    with open(p, "wb") as f:
        et_pm.write_to_file(f)

    print(f"4. io: mkstemp OK ({p})")

    rt = Runtime.get()
    program = rt.load_program(Path(p), verification=Verification.Minimal)
    method = program.load_method("forward")
    print("5. runtime: load_method OK")

    y = method.execute(inputs)
    print("6. runtime: execute OK")
    return y



def main():
    torch.manual_seed(0)
    x = torch.randn(2, 4, dtype=torch.float32, device="cpu")
    input = (x,)
    
    #### Passed test cases
    # run_case("case_nonzero_coreml", Nonzero().eval(), input)
    # run_case("case_nonzero_split_no_squeeze_coreml", NonzeroSplitNoSqueeze().eval(), input)
    # run_case("case_index_bool_unsqueeze_squeeze_f32_coreml", IndexBoolUnsqueezeSqueezeF32().eval(), input)
    
    ##### Failed test cases
    run_case("case_where_1input_coreml", Where1Input().eval(), input)
    # run_case("case_nonzero_numpy_coreml", NonzeroNumpy().eval(), input)
    # run_case("case_nonzero_split_return_row_coreml", NonzeroSplitReturnRow().eval(), input)
    # run_case("case_index_bool_unsqueeze_squeeze_coreml", IndexBoolUnsqueezeSqueeze().eval(), input)

    

if __name__ == "__main__":
    main()
