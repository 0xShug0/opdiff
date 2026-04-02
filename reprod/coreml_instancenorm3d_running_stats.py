"""
Repro summary:

This script compares two otherwise identical CoreML CPU FP32 exports of
`torch.nn.InstanceNorm3d(256, affine=False)`:

- `track_running_stats=False`: export + CoreML inference succeed
- `track_running_stats=True`: export fails when using the default decomposition path

This is intentionally self-contained and does not depend on the opdiff framework.

Suggested run command:
  /Users/leowang/Desktop/Env/env1/bin/python reprod/coreml_instancenorm3d_running_stats.py
"""

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


def run_case(track_running_stats: bool):
    print(f"=== track_running_stats={track_running_stats} ===")

    torch.manual_seed(0)
    x = torch.randn(1, 256, 32, 28, 28, dtype=torch.float32)

    model = nn.InstanceNorm3d(
        256,
        affine=False,
        track_running_stats=track_running_stats,
    ).eval()

    if track_running_stats:
        with torch.no_grad():
            model.running_mean.zero_()
            model.running_var.fill_(1.0)

    model = WrappedModule(model).eval()

    with torch.no_grad():
        y_pt = model(x).detach().cpu().numpy()

    exported = torch.export.export(model, (x,))
    print("1. export: torch.export.export OK")

    exported = exported.run_decompositions()
    print("2. export: run_decompositions() OK")

    try:
        mlmodel = ct.convert(
            exported,
            inputs=[ct.TensorType(shape=x.shape)],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
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
    except Exception as e:
        print(f"3. failure: {type(e).__name__}: {e}")


def main():
    run_case(track_running_stats=False)
    run_case(track_running_stats=True)


if __name__ == "__main__":
    main()
