from dataclasses import dataclass, replace
from typing import Any, Dict
import sys

from backends.onnx_backend import ONNXBackend, ONNXOptions
from backends.torch_backend import TorchBackend, TorchRunOptions

IS_WINDOWS = sys.platform.startswith("win")
IS_DARWIN = sys.platform == "darwin"

if IS_DARWIN:
    import coremltools as ct
    from backends.coreml_backend import CoreMLBackend, CoreMLOptions, TorchExportOptions
    from executorch.runtime import Verification
    from backends.executorch_backend import ExecuTorchBackend, ExecuTorchOptions
else:
    ct = None
    CoreMLBackend = CoreMLOptions = TorchExportOptions = None  # type: ignore
    Verification = None
    ExecuTorchBackend = ExecuTorchOptions = None  # type: ignore
    

@dataclass(frozen=True)
class PresetDelta:
    kind: str  # "torch" | "coreml" | "onnx" | "executorch"
    torch: Dict[str, Any] | None = None
    coreml: Dict[str, Any] | None = None
    torch_export: Dict[str, Any] | None = None
    onnx: Dict[str, Any] | None = None
    executorch: Dict[str, Any] | None = None


BACKEND_PRESETS = {
    # Torch eager
    "torch_cpu_fp32": PresetDelta(kind="torch", torch={"device": "cpu", "run_fp16": False}),
    "torch_cpu_fp16": PresetDelta(kind="torch", torch={"device": "cpu", "run_fp16": True}),
    "torch_cuda_fp32": PresetDelta(kind="torch", torch={"device": "cuda", "run_fp16": False}),
    "torch_cuda_fp16": PresetDelta(kind="torch", torch={"device": "cuda", "run_fp16": True}),
    
    # ONNX
    "onnx_cpu_fp32": PresetDelta(kind="onnx", onnx={"providers": ["CPUExecutionProvider"], "export_fp16": False}),
    "onnx_cuda_fp32": PresetDelta(kind="onnx", onnx={"providers": ["CUDAExecutionProvider"], "export_fp16": False}),
    "onnx_cuda_fp16": PresetDelta(kind="onnx", onnx={"providers": ["CUDAExecutionProvider"], "export_fp16": True}),
}

if IS_DARWIN:
    BACKEND_PRESETS.update({
        # Torch eager
        "torch_mps_fp32": PresetDelta(kind="torch", torch={"device": "mps", "run_fp16": False}),
        "torch_mps_fp16": PresetDelta(kind="torch", torch={"device": "mps", "run_fp16": True}),
    
        # CoreML
        "coreml_cpu_fp32": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.CPU_ONLY, "compute_precision": ct.precision.FLOAT32},
        ),
        
        "coreml_gpu_fp32": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.CPU_AND_GPU, "compute_precision": ct.precision.FLOAT32},
        ),
        
        "coreml_any_fp32": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.ALL, "compute_precision": ct.precision.FLOAT32},
        ),

        # FP16 typed execution (Apple-style): keep PyTorch export in FP32, let CoreML lower to FP16.
        "coreml_cpu_fp16": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.CPU_ONLY, "compute_precision": ct.precision.FLOAT16},
            torch_export={"export_fp16": False},
        ),
        
        "coreml_gpu_fp16": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.CPU_AND_GPU, "compute_precision": ct.precision.FLOAT16},
            torch_export={"export_fp16": False},
        ),
        
        "coreml_any_fp16": PresetDelta(
            kind="coreml",
            coreml={"compute_units": ct.ComputeUnit.ALL, "compute_precision": ct.precision.FLOAT16},
            torch_export={"export_fp16": False},
        ),
        
        # ONNX CoreML EP
        "onnx_coreml_fp32": PresetDelta(kind="onnx", onnx={"providers": ["CoreMLExecutionProvider"], "export_fp16": False}),
        "onnx_coreml_fp16": PresetDelta(kind="onnx", onnx={"providers": ["CoreMLExecutionProvider"], "export_fp16": True}),

        # ExecuTorch
        "executorch_cpu_fp32":    PresetDelta(kind="executorch", executorch={"target": "cpu"}, torch_export={"export_fp16": False}),
        "executorch_cpu_fp16":    PresetDelta(kind="executorch", executorch={"target": "cpu"}, torch_export={"export_fp16": True}),
        "executorch_mps_fp32":    PresetDelta(kind="executorch", executorch={"target": "mps"},    torch_export={"export_fp16": False}),
        "executorch_mps_fp16":    PresetDelta(kind="executorch", executorch={"target": "mps"},    torch_export={"export_fp16": True}),
        "executorch_coreml_fp32": PresetDelta(kind="executorch", executorch={"target": "coreml", "coreml_compute_precision": "fp32"}, torch_export={"export_fp16": False}),
        "executorch_coreml_fp16": PresetDelta(kind="executorch", executorch={"target": "coreml", "coreml_compute_precision": "fp16"}, torch_export={"export_fp16": False}),
    })
    
# ---- base defaults ----
BASE_TORCH_RUN = TorchRunOptions()
BASE_TORCH_EXPORT = TorchExportOptions()

BASE_ONNX = ONNXOptions(
    opset=20,
    providers=["CPUExecutionProvider"],
    fallback_cpu=False,
    dynamo=True,
    do_constant_folding=True,
    dynamic_axes=None,
)

if IS_DARWIN:
    BASE_COREML = CoreMLOptions(
        convert_to="mlprogram",
        minimum_ios_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=None,
    )

    BASE_EXECUTORCH = ExecuTorchOptions(
        target="cpu",
        verification=Verification.Minimal,
    )
else:
    BASE_COREML = None
    BASE_EXECUTORCH = None


def make_backend(
    preset_name: str,
    *,
    # Torch eager overrides
    torch_overrides: dict | None = None,
    # CoreML overrides
    torch_export_overrides: dict | None = None,
    coreml_overrides: dict | None = None,
    # ONNX overrides
    onnx_overrides: dict | None = None,
    # ExecuTorch overrides
    executorch_overrides: dict | None = None,
):
    p = BACKEND_PRESETS[preset_name]

    if p.kind == "torch":
        run_opts = BASE_TORCH_RUN

        # apply preset deltas
        if p.torch:
            run_opts = replace(run_opts, **p.torch)

        # apply runtime overrides
        if torch_overrides:
            run_opts = replace(run_opts, **torch_overrides)

        return TorchBackend(run_opts=run_opts)

    if p.kind == "coreml":
        export_opts = BASE_TORCH_EXPORT
        coreml_opts = BASE_COREML

        if p.torch_export:
            export_opts = replace(export_opts, **p.torch_export)
        if p.coreml:
            coreml_opts = replace(coreml_opts, **p.coreml)

        if torch_export_overrides:
            export_opts = replace(export_opts, **torch_export_overrides)
        if coreml_overrides:
            coreml_opts = replace(coreml_opts, **coreml_overrides)

        return CoreMLBackend(export_opts=export_opts, coreml_opts=coreml_opts)

    if p.kind == "onnx":
        onnx_opts = BASE_ONNX

        if p.onnx:
            onnx_opts = replace(onnx_opts, **p.onnx)
        if onnx_overrides:
            onnx_opts = replace(onnx_opts, **onnx_overrides)

        return ONNXBackend(onnx_opts=onnx_opts)

    if p.kind == "executorch":
        export_opts = BASE_TORCH_EXPORT
        et_opts = BASE_EXECUTORCH

        if p.torch_export:
            export_opts = replace(export_opts, **p.torch_export)
        if p.executorch:
            et_opts = replace(et_opts, **p.executorch)

        if torch_export_overrides:
            export_opts = replace(export_opts, **torch_export_overrides)
        if executorch_overrides:
            et_opts = replace(et_opts, **executorch_overrides)

        return ExecuTorchBackend(export_opts=export_opts, et_opts=et_opts)


    raise ValueError(f"Unknown preset kind: {p.kind}")
