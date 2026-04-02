"""
Repro summary:

For `torch.nn.InstanceNorm3d(256, affine=False, track_running_stats=False)`:

- CoreML CPU FP32 works
- CoreML GPU FP32 crashes
- CoreML ALL FP32 also crashes

This script is self-contained and avoids any opdiff framework dependencies.
It runs each compute-unit configuration in a subprocess so the parent process
can report the crash cleanly.

Suggested run command:
  /Users/leowang/Desktop/Env/env1/bin/python reprod/coreml_instancenorm3d_gpu_crash.py
"""

import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


class WrappedModule(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        x = x.clone()
        return self.mod(x)


def run_child(mode: str):
    compute_units = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "any": ct.ComputeUnit.ALL,
    }[mode]

    torch.manual_seed(0)
    x = torch.randn(1, 256, 32, 28, 28, dtype=torch.float32)

    model = WrappedModule(
        nn.InstanceNorm3d(
            256,
            affine=False,
            track_running_stats=False,
        ).eval()
    ).eval()

    with torch.no_grad():
        y_pt = model(x).detach().cpu().numpy()

    exported = torch.export.export(model, (x,))
    print("1. export: torch.export.export OK")

    exported = exported.run_decompositions()
    print("2. export: run_decompositions() OK")

    mlmodel = ct.convert(
        exported,
        inputs=[ct.TensorType(shape=x.shape)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT32,
    )
    print("3. convert: ct.convert OK")

    inp_name = list(mlmodel.input_description._fd_spec)[0].name
    out_name = list(mlmodel.output_description._fd_spec)[0].name
    y_coreml = mlmodel.predict({inp_name: x.detach().cpu().numpy()})[out_name]
    print("4. runtime: mlmodel.predict OK")

    abs_diff = np.abs(y_pt - y_coreml)
    print("5. diff: max_abs_diff =", float(abs_diff.max()))
    print("5. diff: mean_abs_diff =", float(abs_diff.mean()))


def run_parent():
    for mode in ["cpu", "gpu", "any"]:
        print(f"=== {mode} ===")
        proc = subprocess.run(
            [sys.executable, __file__, "--child", mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("returncode =", proc.returncode)
        if proc.stdout.strip():
            print(proc.stdout.rstrip())
        if proc.stderr.strip():
            print("--- stderr ---")
            print(proc.stderr.rstrip())


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        run_child(sys.argv[2])
    else:
        run_parent()


if __name__ == "__main__":
    main()
