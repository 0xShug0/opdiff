import contextlib
from contextlib import redirect_stdout, redirect_stderr
import glob
import logging
import os
import shutil
import tempfile

import torch
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("opdiff.run")
for name in [
    "onnx_ir",
    "onnxscript",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

logging.getLogger("coremltools").setLevel(logging.ERROR)

@contextlib.contextmanager
def silence_output(enabled: bool = True):
    """Temporarily suppress stdout/stderr (including native/C++ logs) while executing a block."""
    if not enabled:
        yield
        return
    
    # Windows: avoid fd-level dup2 (can break in IDEs/Jupyter/capture setups)
    if os.name == "nt":
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            yield
        return
    
    # POSIX: redirect file descriptors 1 and 2 (catches native/C++ logs too)
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)

def show_dtypes(tag, obj):
    """Recursively print dtype information for tensors within nested Python containers."""
    if torch.is_tensor(obj): print(tag, obj.dtype)
    elif isinstance(obj, (list,tuple)):
        for i,x in enumerate(obj): show_dtypes(f"{tag}[{i}]", x)
    elif isinstance(obj, dict):
        for k,v in obj.items(): show_dtypes(f"{tag}.{k}", v)

def clean_sys_tmp():
    """Remove temporary files/artifacts commonly produced by backend exports and executions."""
    td = os.environ.get("TMPDIR") or tempfile.gettempdir()
    pats = ["onnxruntime-*", "*.mlpackage", "*.mlmodelc", "*.pte", "tmp*", "*.onnx", "*opdiff*"]
    g = lambda: [p for pat in pats for p in glob.glob(os.path.join(td, pat))]
    for p in g():
        try:
            (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
        except FileNotFoundError:
            pass
    for p in [
        "/Users/$USER/Library/Caches/executorchcoreml",
        "/Users/$USER/Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache",
    ]:
        try:
            shutil.rmtree(os.path.expandvars(p))
        except FileNotFoundError:
            pass
               
def assert_out_not_all_zero(out, *, op_name, preset_name):
    """Raise an error if a nested output structure consists entirely of zero-valued elements."""
    def all_zero_leaf(x):
        if isinstance(x, torch.Tensor):
            return x.numel() == 0 or bool((x == 0).all().item())
        if isinstance(x, np.ndarray):
            return x.size == 0 or bool((x == 0).all())
        if isinstance(x, (int, float, bool)):
            return x == 0
        raise TypeError(f"Unsupported out type: {type(x)}")

    def walk(x):
        if isinstance(x, (torch.Tensor, np.ndarray, int, float, bool)):
            return all_zero_leaf(x)
        if isinstance(x, (list, tuple)):
            return all(walk(v) for v in x)
        if isinstance(x, dict):
            return all(walk(v) for v in x.values())
        raise TypeError(f"Unsupported out type: {type(x)}")

    if walk(out):
        raise AssertionError(f"[preset={preset_name} op={op_name}] output is all zeros")