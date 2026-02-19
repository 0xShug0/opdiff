"""
Repro summary:

This script reproduces an ExecuTorch runtime crash during inference when exporting and running a model that
contains aten::prelu with a constant weight embedded in the module (e.g., registered buffer/constant attribute)
and using an explicit empty decomposition table via ExportedProgram.run_decompositions({}).

Observed behavior:
- When the PReLU weight is constant inside the model and run_decompositions({}) is used, ExecuTorch segfaults
  during runtime execution (method.execute), i.e., during inference.
- When the same PReLU weight is provided as a tensor input (non-const), the pipeline completes and inference works.

Expected Behavior:
- ExecuTorch should either execute successfully or raise a Python-visible error if the pattern is unsupported.
  It should not terminate the process with a segmentation fault.
"""


import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import executorch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime, Verification
import warnings
warnings.simplefilter("ignore", FutureWarning)

def run_case(name, model, inputs, decomposition=False):
    print(f"=== {name} ===")
    print(f"run_decompositions: {'default' if decomposition else '{}'}")
    
    exported = torch.export.export(model, args=inputs, strict=True)
    print("1. export: torch.export.export OK")
    
    exported = exported.run_decompositions() if decomposition else exported.run_decompositions({})
    print("2. export: graph")
    exported.graph_module.print_readable()
    
    edge_pm = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=None,
    )
    et_pm = edge_pm.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    print("3. lower: to_executorch OK")
    
    fd, p = tempfile.mkstemp(suffix=".pte", prefix="prelu_")
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


class ConstWPrelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.ones(3, dtype=torch.float32))

    def forward(self, x):
        return torch.ops.aten.prelu.default(x, self.w)


class InputWPrelu(nn.Module): 
    def forward(self, x, w):
        return torch.ops.aten.prelu.default(x, w)


def main():
    torch.manual_seed(0)

    x4d = torch.randn(2, 3, 3, 3, device="cpu", dtype=torch.float32)
    wprelu = torch.ones(3, device="cpu", dtype=torch.float32)
    # Pass
    run_case(
        "case_prelu_input_w",
        InputWPrelu().eval().to(device="cpu", dtype=torch.float32),
        (x4d, wprelu),
    )
    # Pass
    run_case(
        "case_prelu_const_w",
        ConstWPrelu().eval().to(device="cpu", dtype=torch.float32),
        (x4d,),
        decomposition=True,
    )
    # Failed
    run_case(
        "case_prelu_const_w",
        ConstWPrelu().eval().to(device="cpu", dtype=torch.float32),
        (x4d,),
    )

if __name__ == "__main__":
    main()
