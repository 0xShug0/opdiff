"""Runner wrappers for torch ops and user-specified modules.

This module provides small `nn.Module` wrappers used across backends to:

- Resolve a torch operator from a string like "aten::add" (optionally with an overload)
- Resolve and construct modules/functions from a dotted Python path or "file:...py::Symbol"
- Reconstruct full positional/keyword arguments from runtime tensor inputs plus constants
  (including support for "a list of tensors" placeholders via TensorListMarker)
- Sanitize outputs so backends that don't support complex dtypes can still interoperate
"""

import os
import sys
from typing import Any, Dict, List, Optional
import importlib.util
import torch
import torch.nn as nn

from backends.backend_base import Backend, NoopBackend
from core_types import TensorListMarker

def _lookup_torch_op_fn(op: str):
    """Resolve a torch operator string into a callable.

    Supported formats:
    - "namespace::op" (e.g., "aten::add")
    - "namespace::op.overload" (e.g., "aten::add.Tensor")
    """
    try:
        ns, name = op.split("::", 1)
    except ValueError as e:
        raise ValueError(f"Invalid op string: {op}") from e

    namespace = getattr(torch.ops, ns, None)
    if namespace is None:
        raise KeyError(f"torch.ops has no namespace '{ns}'")

    base, dot, overload = name.partition(".")
    base_op = getattr(namespace, base, None)
    if base_op is None:
        raise KeyError(f"torch.ops.{ns} has no op '{base}'")

    if dot:
        fn = getattr(base_op, overload, None)
        if fn is None:
            raise KeyError(f"torch.ops.{ns}.{base} has no overload '{overload}'")
        return fn

    return base_op


def _resolve_op(
    op: str,
    *,
    runner: Backend,
    backend_mode: bool = False,
    cast_input0_to_complex: bool = False,
):
    """Resolve an op and allow the backend to wrap it for export/runtime needs."""
    op_fn = _lookup_torch_op_fn(op)
    op_fn = runner.wrap_op(
        op_fn,
        backend_mode=backend_mode,
        cast_input0_to_complex=cast_input0_to_complex,
    )
    return op_fn


def _resolve_torch_path(path: str):
    """Resolve a Python symbol for module/function construction.

    Supported formats:
    - "torch.nn.Linear"
    - "some.pkg.module.Symbol"
    - "file:/abs/or/rel/path.py::Symbol"

    """
    if path.startswith("file:"):
        spec = path[len("file:"):]
        file_part, sep, sym = spec.partition("::")
        if not sep or not file_part or not sym:
            raise ValueError(f"Invalid file path spec: {path} (expected file:...py::Symbol)")

        file_abspath = os.path.abspath(file_part)

        # Stable module name
        mod_name = f"_opdiff_file_{abs(hash(file_abspath))}"

        mod = sys.modules.get(mod_name)
        if mod is None:
            module_spec = importlib.util.spec_from_file_location(mod_name, file_abspath)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Could not load module from {file_abspath}")

            mod = importlib.util.module_from_spec(module_spec)
            sys.modules[mod_name] = mod
            
            # Temporarily add the module's directory to sys.path to support local imports.
            file_dir = os.path.dirname(file_abspath)
            restore_sys_path = False
            if file_dir and file_dir not in sys.path:
                sys.path.insert(0, file_dir)
                restore_sys_path = True
            try:
                module_spec.loader.exec_module(mod)
            finally:
                if restore_sys_path:
                    # Remove the inserted entry (prefer front removal, fallback to first match).
                    if sys.path and sys.path[0] == file_dir:
                        sys.path.pop(0)
                    else:
                        try:
                            sys.path.remove(file_dir)
                        except ValueError:
                            pass

        obj = getattr(mod, sym, None)
        if obj is None:
            raise AttributeError(f"{path}: symbol '{sym}' not found in {file_abspath}")
        return obj

    # Default: generic dotted python path resolver
    parts = path.split(".")
    if not parts or not parts[0]:
        raise ValueError(f"Invalid python path: {path}")

    # Import the longest module prefix, then getattr the remainder.
    mod = None
    last_err = None
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            mod = __import__(mod_name, fromlist=["*"])
            attr_parts = parts[i:]
            break
        except Exception as e:
            last_err = e
            continue

    if mod is None:
        raise ImportError(f"Could not import any module from '{path}': {last_err}") from last_err

    obj = mod
    for p in attr_parts:
        obj = getattr(obj, p)
    return obj


def _materialize_build_nodes(x: Any, *, resolve_path=_resolve_torch_path) -> Any:
    """Turn ConstructNode/ModuleNode (or raw dict equivalents) into Python objects.

    This is used for module construction specs where constructor args/kwargs may
    themselves contain nested build nodes (including inside lists/dicts).
    """
    t = getattr(x, "type", None)
    if t in ("construct", "module") and hasattr(x, "path"):
        ctor = resolve_path(x.path)
        args = [_materialize_build_nodes(a, resolve_path=resolve_path) for a in (x.args or [])]
        kwargs = {
            k: _materialize_build_nodes(v, resolve_path=resolve_path)
            for k, v in (x.kwargs or {}).items()
        }
        return ctor(*args, **kwargs)

    if isinstance(x, dict):
        t = x.get("type")
        if t in ("construct", "module"):
            ctor = resolve_path(x["path"])
            args = [
                _materialize_build_nodes(a, resolve_path=resolve_path)
                for a in (x.get("args") or [])
            ]
            kwargs = {
                k: _materialize_build_nodes(v, resolve_path=resolve_path)
                for k, v in (x.get("kwargs") or {}).items()
            }
            return ctor(*args, **kwargs)
        return {k: _materialize_build_nodes(v, resolve_path=resolve_path) for k, v in x.items()}

    if isinstance(x, list):
        return [_materialize_build_nodes(v, resolve_path=resolve_path) for v in x]

    return x


def _reconstruct_args_kwargs(
    tensor_args: tuple,
    *,
    const_args: List[Any],
    arg_is_tensor: List[bool],
    const_kwargs: Dict[str, Any],
    kw_tensor_keys: List[str],
):
    """Reconstruct full args/kwargs from runtime tensors plus constants.

    Conventions:
    - `arg_is_tensor` describes each positional arg in order:
        True  -> take next item from `tensor_args`
        False -> take next item from `const_args`
    - If a constant positional arg is a TensorListMarker(n), consume the next n
      tensors from `tensor_args` and pass them as a Python list.
    - `kw_tensor_keys` consumes additional tensor args (in order) and inserts them
      into kwargs under those names.
    """
    t_i = 0
    c_i = 0
    full_args: List[Any] = []

    for is_t in arg_is_tensor:
        if is_t:
            full_args.append(tensor_args[t_i])
            t_i += 1
        else:
            c = const_args[c_i]
            c_i += 1
            if isinstance(c, TensorListMarker):
                n = c.n
                full_args.append(list(tensor_args[t_i : t_i + n]))
                t_i += n
            else:
                full_args.append(c)

    full_kwargs = dict(const_kwargs)
    for k in kw_tensor_keys:
        full_kwargs[k] = tensor_args[t_i]
        t_i += 1

    return full_args, full_kwargs



    
class TorchOpRunner(nn.Module):
    """Eager runner for a torch op resolved from a "namespace::op[.overload]" string."""
     
    def __init__(
        self,
        op: str,
        *,
        cast_input0_to_complex: bool = False,
        runner: Backend | None = None,
    ):
        super().__init__()
        self.runner = runner or NoopBackend()
        self.op = _resolve_op(
            op,
            runner=self.runner,
            backend_mode=False,
            cast_input0_to_complex=cast_input0_to_complex,
        )

    def forward(self, *args, **kwargs):
        y = self.op(*args, **kwargs)
        return y


class BackendOpRunner(nn.Module):
    """Export-friendly runner for a torch op with mixed tensor/constant args."""
    
    def __init__(
        self,
        op: str,
        const_args: List[Any],
        arg_is_tensor: List[bool],
        *,
        const_kwargs: Optional[Dict[str, Any]] = None,
        kw_tensor_keys: Optional[List[str]] = None,
        cast_input0_to_complex: bool = False,
        runner: Backend | None = None,
    ):
        super().__init__()
        self.runner = runner or NoopBackend()
        self.op = _resolve_op(
            op,
            runner=self.runner,
            backend_mode=True,
            cast_input0_to_complex=cast_input0_to_complex,
        )
        self.const_args = const_args
        self.arg_is_tensor = arg_is_tensor
        self.const_kwargs = const_kwargs or {}
        self.kw_tensor_keys = kw_tensor_keys or []

    def forward(self, *tensor_args):
        full_args, full_kwargs = _reconstruct_args_kwargs(
            tensor_args,
            const_args=self.const_args,
            arg_is_tensor=self.arg_is_tensor,
            const_kwargs=self.const_kwargs,
            kw_tensor_keys=self.kw_tensor_keys,
        )
        y = self.op(*full_args, **full_kwargs)

        # if isinstance(y, (tuple, list)):
        #     y = y[0]

        return self.runner.sanitize_output(y)


class TorchModuleRunner(nn.Module):
    """Instantiate and run a module/function referenced by a path spec.

    The referenced object can be:
    - An nn.Module class: instantiated once in __init__
    - A callable factory that returns an nn.Module: called once in __init__
    - A plain callable: invoked each forward call

    Constructor args/kwargs may include nested build nodes (or raw dict equivalents),
    which are materialized before instantiation.
    """

    def __init__(self, path: str, args: list, kwargs: dict):
        super().__init__()
        self.path = path
        self.fn = _resolve_torch_path(path)

        # Materialize nested nodes in constructor args/kwargs
        self.args = [_materialize_build_nodes(a) for a in (args or [])]
        self.kwargs = {k: _materialize_build_nodes(v) for k, v in (kwargs or {}).items()}
        self.mod = None
        if isinstance(self.fn, type) and issubclass(self.fn, nn.Module):
            self.mod = self.fn(*self.args, **self.kwargs)
            self.mod.eval()
        elif callable(self.fn):
            maybe_mod = self.fn(*self.args, **self.kwargs)
            if isinstance(maybe_mod, nn.Module):
                self.mod = maybe_mod
                self.mod.eval()

    def forward(self, *inputs, **forward_kwargs):
        if self.mod is not None:
            return self.mod(*inputs, **(forward_kwargs or {}))

        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(forward_kwargs or {})
        return self.fn(*inputs, *self.args, **merged_kwargs)


class BackendModuleRunner(nn.Module):
    """Export-friendly wrapper for a module spec.

    This wrapper:
    - Instantiates the module once (via TorchModuleRunner)
    - Reconstructs mixed tensor/constant args at runtime
    - Sanitizes outputs through the backend runner
    """

    def __init__(
        self,
        mod_spec,
        *,
        const_args,
        arg_is_tensor,
        const_kwargs,
        kw_tensor_keys,
        runner,
    ):
        super().__init__()
        self.runner = runner
        self.const_args = const_args or []
        self.arg_is_tensor = arg_is_tensor or []
        self.const_kwargs = const_kwargs or {}
        self.kw_tensor_keys = kw_tensor_keys or []

        self.mod = TorchModuleRunner(mod_spec.path, mod_spec.args, mod_spec.kwargs)
        self.mod.eval()

    def forward(self, *tensor_args):
        full_args, full_kwargs = _reconstruct_args_kwargs(
            tensor_args,
            const_args=self.const_args,
            arg_is_tensor=self.arg_is_tensor,
            const_kwargs=self.const_kwargs,
            kw_tensor_keys=self.kw_tensor_keys,
        )
        y = self.mod(*full_args, **full_kwargs)
        return self.runner.sanitize_output(y)
