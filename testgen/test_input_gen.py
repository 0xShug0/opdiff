"""Input generation and export preparation for op tests.

This module provides `InputGen`, a deterministic sampler that produces runtime inputs
matching a declarative `Spec` tree (TensorSpec, ScalarSpec, ListSpec, OptionalSpec, etc.).

It also prepares an "export bundle" representation used by export backends:
- Positional args are split into tensor-only args plus constant args (with markers for tensor-lists)
- Keyword args are similarly split into constant kwargs plus an ordered list of tensor kwarg keys

Export-oriented normalization rules:
- Some exporters/backends expect rank-1 tensors for scalar-like values; this module
  normalizes rank-0 tensors/scalars accordingly (see `to_export_value`).
- Complex tensors are sampled as complex when requested by specs; backends may later
  pack/convert complex representations depending on their capabilities.

Determinism:
- Sampling is deterministic with respect to `InputGen.seed`. Batch sampling methods
  either share a single RNG stream (sample_many) or use one seed per batch (sample_many_k).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from core_types import INT_DTYPES, ConstSpec, ConstTensorSpec, ListSpec, OptionalSpec, Precision, ScalarSpec, ScalarTensorSpec, Spec, TensorSpec
from testgen.test_plan import build_specs, split_export_args


class InputGen:

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        
    # ---- RNG ----
    def _with_seed(self, seed: int, fn):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return fn()
        
    # ---- Export ----
    def to_export_value(self, value: Any, spec: Spec, device_hint: Optional[torch.Tensor] = None) -> Any:
        """Convert a sampled runtime value into the representation expected by export/backends for the given spec.

        This function normalizes scalar vs tensor inputs and applies export-specific shape rules.

        Args:
            value: The sampled value produced by `sample()` / `sample_many()`. Can be a torch.Tensor, a Python
                scalar (int/float/bool), or a nested container (list/tuple/dict) matching `spec`.
            spec: The Spec describing the expected structure and dtype/shape semantics.
            device_hint: Optional tensor used only as a device hint when creating new tensors from Python
                scalars (e.g., for ScalarSpec/ScalarTensorSpec). If None, tensors are created on CPU.

        Returns:
            A value structurally matching `spec`, where:
            - TensorSpec: returns a torch.Tensor with dtype enforced (if specified) and rank-0 reshaped to (1,).
            - ScalarSpec: returns a torch.Tensor of shape (1,) (rank-1) on `device_hint.device` (or CPU).
            - ScalarTensorSpec: returns a rank-0 torch.Tensor on `device_hint.device` (or CPU).
            - ListSpec / OptionalSpec / ConstTensorSpec: recursively converted elements.
            - Other values are returned unchanged.

        Notes:
            - This is intended for export preparation; it may convert Python scalars into tensors to align with
            exporter/backend signature expectations.
            - Only uses `device_hint` for device placement during scalar→tensor creation; backends may still
            move tensors to their target device later.
        """
        device = device_hint.device if device_hint is not None else torch.device("cpu")

        if isinstance(spec, TensorSpec):
            if not torch.is_tensor(value):
                raise TypeError("TensorSpec expects a torch.Tensor value")
            value = self._ensure_dtype(value, spec.dtype)
            return self._export_rank0_as_rank1(value)

        if isinstance(spec, ScalarSpec):
            if torch.is_tensor(value):
                return self._export_rank0_as_rank1(value)
            dt = spec.dtype or self._fallback_dtype_from_kind(spec.kind)
            return self._export_scalar_as_rank1_tensor(value, device=device, dtype=dt)

        if isinstance(spec, ScalarTensorSpec):
            if torch.is_tensor(value):
                return self._ensure_dtype(value, spec.dtype)
            dt = spec.dtype or self._fallback_dtype_from_kind(spec.kind)
            return torch.tensor(value, device=device, dtype=dt)

        if isinstance(spec, ListSpec):
            return [self.to_export_value(v, spec.elem, device_hint=device_hint) for v in value]

        if isinstance(spec, OptionalSpec):
            return None if value is None else self.to_export_value(value, spec.elem, device_hint=device_hint)

        if isinstance(spec, ConstTensorSpec):
            return self.to_export_value(value, spec.tensor, device_hint=device_hint)

        return value

    def to_export_values(self, values: Sequence[Any], specs: Sequence[Spec]) -> List[Any]:
        device_anchor = next((v for v in values if torch.is_tensor(v)), None)
        return [self.to_export_value(v, s, device_hint=device_anchor) for v, s in zip(values, specs)]

    def to_export_kwargs(self, kw_values: Dict[str, Any], kw_specs: Dict[str, Spec]) -> Dict[str, Any]:
        if not kw_specs:
            return {}
        device_anchor = next((v for v in kw_values.values() if torch.is_tensor(v)), None)
        return {k: self.to_export_value(kw_values[k], kw_specs[k], device_hint=device_anchor) for k in kw_specs.keys()}

    # ---- Spec materialization ----
    def _apply_precision_if_missing_dtype(self, spec: Any, precision: Precision) -> Any:
        if hasattr(spec, "dtype") and getattr(spec, "dtype") is None and hasattr(spec, "kind"):
            return replace(spec, dtype=getattr(precision, getattr(spec, "kind")))
        return spec

    def _materialize_spec(self, spec: Spec, precision: Optional[Precision]) -> Spec:
        if precision is None:
            return spec

        spec = self._apply_precision_if_missing_dtype(spec, precision)

        if isinstance(spec, ListSpec):
            return ListSpec(elem=self._materialize_spec(spec.elem, precision), length=spec.length)

        if isinstance(spec, OptionalSpec):
            return OptionalSpec(elem=self._materialize_spec(spec.elem, precision), none_prob=spec.none_prob)

        if isinstance(spec, ConstTensorSpec):
            return ConstTensorSpec(tensor=self._materialize_spec(spec.tensor, precision), value=spec.value)

        return spec

    # ---- Export helpers ----
    def _fallback_dtype_from_kind(self, kind: str) -> torch.dtype:
        if kind == "bool":
            return torch.bool
        if kind == "float":
            return torch.float32
        return torch.int64

    def _ensure_dtype(self, t: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
        return t if (dtype is None or t.dtype == dtype) else t.to(dtype)

    def _export_rank0_as_rank1(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(1) if t.ndim == 0 else t

    def _export_scalar_as_rank1_tensor(self, value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.tensor(value, device=device, dtype=dtype)
        return t.reshape(1) if t.ndim == 0 else t

    # ---- Sampling ----
    def sample(self, spec: Spec, precision: Optional[Precision] = None) -> Any:
        """
        Sample a single value matching spec. If precisionis provided, 
        fill missing dtypes from it (e.g., float/int/bool) before sampling. 
        Deterministic w.r.t.self.seed.
        """
        spec2 = self._materialize_spec(spec, precision)
        return self._with_seed(self.seed, lambda: self._sample(spec2))

    def sample_k(self, spec: Spec, k: int, precision: Optional[Precision] = None) -> List[Any]:
        """Sample k independent values for the samespec. Deterministic w.r.t.self.seed."""
        if k <= 0:
            raise ValueError("k must be > 0")
        spec2 = self._materialize_spec(spec, precision)
        return [self._with_seed(self.seed + i, lambda s=spec2: self._sample(s)) for i in range(k)]
    
    def sample_many(self, specs: Sequence[Spec], precision: Optional[Precision] = None) -> List[Any]:
        """Sample one batch of values for a list of specs using a single RNG stream.

        Args:
            specs: Sequence of Spec objects to sample in order.
            precision: Optional Precision used to fill missing dtypes in specs before sampling.

        Returns:
            A list of sampled values aligned with `specs` (same length, same order).

        Determinism:
            Uses `self.seed` once to seed the RNG for the whole batch. Values are deterministic for a
            fixed seed and spec list, but changing the spec list/order changes later samples because all
            draws share the same RNG stream.

        Example:
            specs = [A, B]
            gen = InputGen(seed=0)
            gen.sample_many(specs)  -> [A0, B0]
            gen.sample_many(specs)  -> [A0, B0]   # same again (seed reused)
        """
        specs2 = [self._materialize_spec(s, precision) for s in specs]
        return self._with_seed(self.seed, lambda: [self._sample(s) for s in specs2])

    def sample_many_k(self, specs: Sequence[Spec], k: int, precision: Optional[Precision] = None) -> List[List[Any]]:
        """Sample k batches of values for the same list of specs, with one RNG seed per batch.

        Args:
            specs: Sequence of Spec objects to sample in order for each batch.
            k: Number of batches to generate (must be > 0).
            precision: Optional Precision used to fill missing dtypes in specs before sampling.

        Returns:
            A list of length `k`. Element at index `i` contains the sampled values for `specs`
            using RNG seed `self.seed + i`.
        Determinism:
            Batch `i` uses seed `self.seed + i`, so batches are reproducible and stable by index.

        Example:
            specs = [A, B]
            gen = InputGen(seed=0)
            gen.sample_many_k(specs, 3) -> [[A0, B0], [A1, B1], [A2, B2]]
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        specs2 = [self._materialize_spec(s, precision) for s in specs]
        return [
            self._with_seed(self.seed + i, lambda ss=specs2: [self._sample(s) for s in ss])
            for i in range(k)
        ]
        
    def _sample(self, spec: Spec) -> Any:
        if isinstance(spec, TensorSpec):
            return self._tensor(spec)

        if isinstance(spec, ScalarSpec):
            return self._scalar(spec)

        if isinstance(spec, ScalarTensorSpec):
            return self._scalar_tensor(spec)

        if isinstance(spec, ListSpec):
            return [self._sample(spec.elem) for _ in range(spec.length)]

        if isinstance(spec, OptionalSpec):
            if spec.none_prob <= 0:
                return self._sample(spec.elem)
            return None if torch.rand((), device="cpu").item() < spec.none_prob else self._sample(spec.elem)

        if isinstance(spec, ConstSpec):
            return spec.value

        if isinstance(spec, ConstTensorSpec):
            if spec.value is not None:
                if spec.tensor.dtype is None:
                    raise ValueError("ConstTensorSpec.tensor.dtype is None; pass precision or set dtype explicitly")
                t = torch.tensor(spec.value, dtype=spec.tensor.dtype)
                return t.reshape(spec.tensor.shape)
            return self._tensor(spec.tensor)

        raise TypeError(f"Unknown spec type: {type(spec)}")

    def _auto_dist(self, dtype: torch.dtype) -> str:
        if dtype is torch.bool:
            return "bernoulli"
        if dtype.is_floating_point or dtype.is_complex:
            return "normal"
        return "randint"

    def _sample_tensor_by_dist(
        self,
        *,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        dist: str,
        low: int,
        high: int,
        mean: float,
        std: float,
        p: float,
    ) -> torch.Tensor:
        if dist == "zeros":
            return torch.zeros(shape, device=device, dtype=dtype)

        if dist == "ones":
            return torch.ones(shape, device=device, dtype=dtype)

        if dist == "bernoulli":
            if dtype is not torch.bool and not dtype.is_floating_point:
                raise ValueError("bernoulli requires bool or floating dtype")
            probs = torch.full(shape, float(p), device=device, dtype=torch.float32)
            t = torch.bernoulli(probs)
            return t.to(torch.bool) if dtype is torch.bool else t.to(dtype)

        if dist == "randint":
            if dtype not in INT_DTYPES:
                raise ValueError("randint requires integer dtype")
            return torch.randint(int(low), int(high), shape, device=device, dtype=dtype)

        if dist == "uniform":
            if not (dtype.is_floating_point or dtype.is_complex):
                raise ValueError("uniform requires floating dtype")
            if dtype.is_complex:
                base = torch.float32 if dtype == torch.complex64 else torch.float64
                real = torch.empty(shape, device=device, dtype=base).uniform_(float(low), float(high))
                imag = torch.empty(shape, device=device, dtype=base).uniform_(float(low), float(high))
                return torch.complex(real, imag)
            return torch.empty(shape, device=device, dtype=dtype).uniform_(float(low), float(high))

        if dist == "normal":
            if not (dtype.is_floating_point or dtype.is_complex):
                raise ValueError("normal requires floating dtype")
            if dtype.is_complex:
                base = torch.float32 if dtype == torch.complex64 else torch.float64
                real = torch.empty(shape, device=device, dtype=base).normal_(float(mean), float(std))
                imag = torch.empty(shape, device=device, dtype=base).normal_(float(mean), float(std))
                return torch.complex(real, imag)
            return torch.empty(shape, device=device, dtype=dtype).normal_(float(mean), float(std))

        raise ValueError(f"Unsupported dist: {dist}")

    def _tensor(self, s: TensorSpec) -> torch.Tensor:
        if any(d < 0 for d in s.shape):
            raise ValueError(f"Negative dim in shape: {s.shape}")
        if s.dtype is None:
            raise ValueError("TensorSpec.dtype is None; pass precision or set dtype explicitly")

        device = torch.device(s.device)
        dtype = s.dtype
        dist = s.dist or self._auto_dist(dtype)

        t = self._sample_tensor_by_dist(
            shape=s.shape,
            device=device,
            dtype=dtype,
            dist=dist,
            low=s.low,
            high=s.high,
            mean=s.mean,
            std=s.std,
            p=s.p,
        )

        t.requires_grad_(bool(s.requires_grad) and t.is_floating_point())
        return t

    def _scalar_tensor(self, s: ScalarTensorSpec) -> torch.Tensor:
        if s.dtype is None:
            raise ValueError("ScalarTensorSpec.dtype is None; pass precision or set dtype explicitly")

        device = torch.device(s.device)
        dtype = s.dtype

        if s.value is not None:
            return torch.tensor(s.value, device=device, dtype=dtype)

        dist = s.dist or self._auto_dist(dtype)
        return self._sample_tensor_by_dist(
            shape=(),
            device=device,
            dtype=dtype,
            dist=dist,
            low=s.low,
            high=s.high,
            mean=s.mean,
            std=s.std,
            p=s.p,
        )

    def _scalar(self, s: ScalarSpec) -> Any:
        if s.choices is not None:
            if len(s.choices) == 0:
                raise ValueError("Choices cannot be empty")
            idx = int(torch.randint(0, len(s.choices), ()).item())
            return s.choices[idx]

        if s.kind == "int":
            return int(torch.randint(int(s.low), int(s.high) + 1, ()).item())

        if s.kind == "bool":
            return bool(torch.randint(0, 2, ()).item())

        if s.kind == "float":
            return float(torch.rand(()).item() * (s.high - s.low) + s.low)

        raise ValueError(f"Unsupported scalar kind: {s.kind}")

    def cast_precision(self, x, *, dtype=None):
        """Recursively cast floating-point tensors inside a nested structure to a target dtype.

        Args:
            x: Arbitrary nested structure containing torch.Tensors and/or Python containers
                (list, tuple, dict). Non-tensor leaf values are allowed.
            dtype: Target torch floating dtype (e.g., torch.float16, torch.float32). If None, `x`
                is returned unchanged.

        Returns:
            A structure with the same shape/type as `x`, where any torch.Tensor that is floating-point
            and not already `dtype` is converted via `tensor.to(dtype=dtype)`. Integer, boolean, and
            complex tensors are left unchanged.

        Notes:
            - Only affects tensors for which `torch.is_floating_point(tensor)` is True.
            - Container types are preserved: lists stay lists, tuples stay tuples, dict keys unchanged.
        """
        import torch

        if dtype is None:
            return x

        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x) and x.dtype != dtype:
                return x.to(dtype=dtype)
            return x

        if isinstance(x, list):
            return [self.cast_precision(v, dtype=dtype) for v in x]
        if isinstance(x, tuple):
            return tuple(self.cast_precision(v, dtype=dtype) for v in x)
        if isinstance(x, dict):
            return {k: self.cast_precision(v, dtype=dtype) for k, v in x.items()}

        return x

    def prepare_test_run(
        self,
        cfg,
        test: dict,
        *,
        precision: Optional[Precision] = None,
        k: int = 1,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare deterministic eager inputs and export bundles for a batched op test.

        This function generates `k` independent, reproducible input sets for a single op
        test specification. Each input set includes:
        - eager inputs for direct PyTorch execution, and
        - a corresponding export bundle that separates tensor inputs from constant inputs
          and records structural metadata required by export backends.

        All `k` input sets share the same spec structure and differ only by their deterministic
        sampling index.

        Args:
            cfg: Parsed test configuration used to build input specs (dims, presets, etc.).
            test: Test description dictionary
            precision: Optional Precision used to fill missing dtypes during sampling and to
                cast floating-point tensors for consistency across backends.
            k: Number of independent input sets to generate (must be > 0).

        Returns:
            A tuple `(op_name, eager_inputs_k, export_bundle_k)`:
            - op_name: The op identifier from the test spec.
            - eager_inputs_k: List of length `k`, where each element is a dict with keys:
                - "args": positional eager inputs
                - "kwargs": keyword eager inputs
            - export_bundle_k: List of length `k`, where each element is a dict describing
              tensor arguments, constant arguments, keyword structure, and export metadata
              for backend execution.

        Determinism:
            Input set `i` is sampled using RNG seed `self.seed + i`. For a fixed seed, spec,
            and precision, all generated eager inputs and export bundles are reproducible
            and stable by index.
            
        Notes:
            - The export bundle is structured so export backends can reconstruct the original call:
              tensor inputs are separated from constant (Python) values, and tensor-list positions are
              represented with markers in the constant argument list.
            - Scalar-like values may be normalized to exporter-friendly tensor shapes (e.g., rank-0 to rank-1)
              to match backend signature expectations.
            - If `precision` is provided, missing dtypes in specs are filled from it, and floating-point tensors
              in the returned eager inputs and export bundles are cast to `precision.float` for consistency.
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        
        # Default: float32
        if precision is None:
            precision = Precision(float=torch.float32)

        op_name = test["op"]
        pos_specs, kwarg_specs = build_specs(cfg, test)

        # Positional args: k batches
        pos_args_eager_k = self.sample_many_k(pos_specs, k, precision=precision)

        # Kwargs: sample in a stable key order
        kw_keys = list(kwarg_specs.keys()) if kwarg_specs else []
        kw_specs_list = [kwarg_specs[kk] for kk in kw_keys] if kw_keys else []
        kw_vals_k = self.sample_many_k(kw_specs_list, k, precision=precision) if kw_specs_list else [[] for _ in range(k)]

        eager_inputs_k: List[Dict[str, Any]] = []
        export_bundle_k: List[Dict[str, Any]] = []
        _dtype = precision.float

        for i in range(k):
            pos_args_eager = pos_args_eager_k[i]
            pos_args_export = self.to_export_values(pos_args_eager, pos_specs)

            kwargs_eager = {kk: vv for kk, vv in zip(kw_keys, kw_vals_k[i])} if kw_keys else {}
            kwargs_export = self.to_export_kwargs(kwargs_eager, kwarg_specs) if kwarg_specs else {}

            (
                export_tensor_args,
                export_const_args,
                export_arg_is_tensor,
                export_const_kwargs,
                export_kw_tensor_keys,
            ) = split_export_args(
                pos_args_export,
                pos_specs,
                kwargs_values=kwargs_export,
                kwargs_specs=kwarg_specs,
            )

            eager_inputs = {
                "args": pos_args_eager,
                "kwargs": kwargs_eager,
            }
            export_bundle = {
                "tensor_args": export_tensor_args,
                "const_args": export_const_args,
                "arg_is_tensor": export_arg_is_tensor,
                "const_kwargs": export_const_kwargs,
                "kw_tensor_keys": export_kw_tensor_keys,
                "device": test["device"],
                "cast_input0_to_complex": test.get("cast_input0_to_complex", False),
            }

            eager_inputs_k.append(self.cast_precision(eager_inputs, dtype=_dtype))
            export_bundle_k.append(self.cast_precision(export_bundle, dtype=_dtype))

        return op_name, eager_inputs_k, export_bundle_k