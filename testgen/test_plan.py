"""
Node resolution, Spec construction, and export helpers for op tests.

This module converts validated YAML input nodes into internal `Spec` objects used
for deterministic input generation and backend export.

It handles:
- Resolving preset references (`RefNode`) and symbolic dimensions/ranges.
- Lowering node models (tensor, scalar, list, optional, const) into sampling Specs
  with concrete shapes and dtypes.
- Flattening test configurations into normalized dictionaries.
- Splitting positional and keyword arguments into tensor inputs vs constants for
  export backends.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from core_types import DEFAULT_DTYPE_BY_KIND, DTYPE_MAP, TensorListMarker
from core_types import ConstSpec, ConstTensorSpec, ListSpec, OptionalSpec, ScalarSpec, ScalarTensorSpec, Spec, TensorSpec


from testgen.test_validator import (
    Config,
    PairTest,
    RefNode,
    TensorNode,
    ScalarNode,
    IntListNode,
    ListNode,
    TupleNode,
    OptionalNode,
    ConstNode,
    ConstTensorNode,
    ScalarTensorNode,   
    Node,
)



# -------------------------
# Node resolving
# -------------------------

def resolve_node(node: Node, presets: Dict[str, Node]) -> Node:
    """Resolve preset references inside an input-node tree.

    This function expands `RefNode(ref="...")` by looking up the referenced preset in `presets`,
    and then recursively resolves any nested refs inside the preset value. It also recursively
    resolves the element types of container nodes (e.g., list/tuple/optional). Leaf nodes
    (e.g., tensor/scalar/const nodes) are returned unchanged.

    Args:
        node: The input node to resolve. May contain `RefNode` and/or nested container nodes.
        presets: Mapping from preset name to node definition used to expand `RefNode`.

    Returns:
        A node equivalent to `node` with all preset references expanded and all nested nodes
        recursively resolved.

    Raises:
        KeyError: If a `RefNode` refers to a preset name not present in `presets`.
    """
    if isinstance(node, RefNode):
        if node.ref not in presets:
            raise KeyError(f"Unknown preset ref: {node.ref}")
        return resolve_node(presets[node.ref], presets)

    if isinstance(node, ListNode):
        return ListNode(type="list", len=node.len, elem=resolve_node(node.elem, presets))

    if isinstance(node, TupleNode):
        return TupleNode(
            type="tuple",
            elems=[resolve_node(e, presets) for e in node.elems],
        )

    if isinstance(node, OptionalNode):
        return OptionalNode(
            type="optional",
            p_none=node.p_none,
            elem=resolve_node(node.elem, presets),
        )

    return node


def resolve_shape(shape: List[Any], dims: Dict[str, int]) -> Tuple[int, ...]:
    """Resolve a symbolic shape into a concrete integer shape.

    Each element of `shape` may be:
    - an int (used as-is), or
    - a string symbol (looked up in `dims`).

    Args:
        shape: A list of dimension entries (ints or symbolic strings).
        dims: Mapping from symbolic dimension names to concrete integer values.

    Returns:
        A tuple of ints representing the resolved shape.

    Raises:
        KeyError: If a symbolic dimension in `shape` is not present in `dims`.
    """
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        else:
            if d not in dims:
                raise KeyError(f"Unknown dim symbol: {d}")
            out.append(int(dims[d]))
    return tuple(out)



def resolve_rnge(rnge_value: Any, dims: Dict[str, int], *, as_int: bool = False) -> Any:
    """Resolve a range parameter that may be numeric, symbolic, or a numeric string.

    Args:
        rnge_value: The raw range value (None, int/float, symbol string like "V", or numeric string like "10").
        dims: Mapping from symbolic names to concrete integer values.
        as_int: If True, return an int; otherwise return a float (for numeric inputs).

    Returns:
        - None if rnge_value is None
        - int/float for numeric inputs (including numeric strings)
        - int/float for symbolic strings found in `dims`
        - rnge_value unchanged for unrecognized non-string types

    Raises:
        KeyError: If rnge_value is a string that is neither a key in `dims` nor parseable as a number.
    """
    if rnge_value is None:
        return None
    if isinstance(rnge_value, (int, float)):
        return int(rnge_value) if as_int else float(rnge_value)
    if isinstance(rnge_value, str):
        s = rnge_value.strip()
        if s in dims:
            return int(dims[s]) if as_int else float(dims[s])
        # allow numeric strings
        try:
            return int(s) if as_int else float(s)
        except Exception:
            raise KeyError(f"Unknown symbol (or non-numeric) range value: {rnge_value}")
    return rnge_value

# -------------------------
# Node → Spec lowering
# -------------------------

def _dtype_from_tensor_node(n: TensorNode) -> torch.dtype:
    if getattr(n, "dtype", None):
        return DTYPE_MAP[n.dtype]
    k = getattr(n, "kind", None)
    if not k:
        raise ValueError("TensorNode missing both dtype and kind")
    if k not in DEFAULT_DTYPE_BY_KIND:
        raise ValueError(f"Unsupported tensor kind '{k}'")
    return DEFAULT_DTYPE_BY_KIND[k]


def _scalar_kind_and_dtype(n: ScalarNode) -> Tuple[str, torch.dtype]:
    if getattr(n, "dtype", None):
        dt_str = n.dtype
        if dt_str == "bool":
            return "bool", torch.bool
        if dt_str.startswith("int") or dt_str.startswith("uint"):
            return "int", DTYPE_MAP[dt_str]
        return "float", DTYPE_MAP[dt_str]

    k = getattr(n, "kind", None)
    if not k:
        raise ValueError("ScalarNode missing both dtype and kind")
    if k not in DEFAULT_DTYPE_BY_KIND:
        raise ValueError(f"Unsupported scalar kind '{k}'")
    return k, DEFAULT_DTYPE_BY_KIND[k]

def node_to_spec(node: Node, dims: Dict[str, int]) -> Spec:
    """Lower a validated YAML input node into an internal sampling Spec with concrete shapes and dtypes.

    This converts parsed/validated node models (e.g., TensorNode, ScalarNode, ListNode) into Spec objects
    used by the input generator. Symbolic shape entries and symbolic range parameters are resolved using
    `dims`.

    Args:
        node: A validated input node (or an already-constructed Spec). May contain symbolic dims/ranges.
        dims: Mapping from symbolic dimension names to concrete integer values.

    Returns:
        A Spec (or nested Spec structure) suitable for sampling, where:
        - TensorNode / ConstTensorNode -> TensorSpec / ConstTensorSpec with concrete `shape` and torch dtype
        - ScalarNode -> ScalarSpec with kind/dtype and optional fixed choice
        - ScalarTensorNode -> ScalarTensorSpec (rank-0 tensor spec)
        - ListNode / OptionalNode / TupleNode -> corresponding nested spec structure
        - RefNode is not handled here (should be resolved first via `resolve_node`)

    Raises:
        KeyError: If a symbolic dim or symbolic range value is not found in `dims`.
        ValueError: If required fields are missing or invalid (e.g., unsupported kind/dtype).
        TypeError: If `node` is an unsupported node type.
    """

    if isinstance(node, (TensorSpec, ScalarSpec, ListSpec, OptionalSpec, ConstSpec)):
        return node

    if isinstance(node, TensorNode):
        dist = node.init
        want_int = (dist == "randint")
        return TensorSpec(
            shape=resolve_shape(node.shape, dims),
            kind=getattr(node, "kind", "float"),
            dtype=_dtype_from_tensor_node(node),
            dist=node.init,
            low=resolve_rnge(node.low, dims, as_int=want_int),
            high=resolve_rnge(node.high, dims, as_int=want_int),
            mean=float(resolve_rnge(node.mean, dims)),
            std=float(resolve_rnge(node.std, dims)),
            p=float(resolve_rnge(node.p, dims)),
            requires_grad=node.requires_grad,
        )

    if isinstance(node, ScalarNode):
        k, dt = _scalar_kind_and_dtype(node)
        if getattr(node, "value", None) is not None:
            return ScalarSpec(kind=k, choices=(node.value,), dtype=dt)

        return ScalarSpec(
            kind=k,
            low=float(resolve_rnge(node.low, dims)) if node.low is not None else ScalarSpec.low,
            high=float(resolve_rnge(node.high, dims)) if node.high is not None else ScalarSpec.high,
            choices=None,
            dtype=dt,
        )

    if isinstance(node, ScalarTensorNode):
        dt = DTYPE_MAP[node.dtype] if getattr(node, "dtype", None) else DEFAULT_DTYPE_BY_KIND[getattr(node, "kind", "float")]
        k = getattr(node, "kind", None)
        if k is None:
            if dt is torch.bool:
                k = "bool"
            elif dt.is_floating_point or dt.is_complex:
                k = "float"
            else:
                k = "int"

        return ScalarTensorSpec(
            kind=k,
            dtype=dt,
            dist=getattr(node, "init", None),
            low=float(resolve_rnge(node.low, dims)) if node.low is not None else ScalarSpec.low,
            high=float(resolve_rnge(node.high, dims)) if node.high is not None else ScalarSpec.high,
            mean=float(getattr(node, "mean", 0.0)),
            std=float(getattr(node, "std", 1.0)),
            p=float(getattr(node, "p", 0.5)),
            value=getattr(node, "value", None),
        )

    if isinstance(node, IntListNode):
        return ConstSpec(value=tuple(node.elems))

    if isinstance(node, ListNode):
        return ListSpec(elem=node_to_spec(node.elem, dims), length=int(resolve_rnge(node.len, dims)))

    if isinstance(node, OptionalNode):
        return OptionalSpec(elem=node_to_spec(node.elem, dims), none_prob=node.p_none)

    if isinstance(node, TupleNode):
        return tuple(node_to_spec(e, dims) for e in node.elems)

    if isinstance(node, ConstNode):
        return ConstSpec(value=node.value)

    if isinstance(node, ConstTensorNode):
        dist = node.init
        want_int = (dist == "randint")
        t = TensorSpec(
            shape=resolve_shape(node.shape, dims),
            kind=getattr(node, "kind", "float"),
            dtype=_dtype_from_tensor_node(node),
            dist=node.init,
            low=resolve_rnge(node.low, dims, as_int=want_int),
            high=resolve_rnge(node.high, dims, as_int=want_int),
            mean=float(resolve_rnge(node.mean, dims)),
            std=float(resolve_rnge(node.std, dims)),
            p=float(resolve_rnge(node.p, dims)),
            requires_grad=False,
        )
        return ConstTensorSpec(tensor=t, value=getattr(node, "value", None))

    raise TypeError(f"Unhandled node type: {type(node)}")


# -------------------------
# Test extraction
# -------------------------

def _testcase_to_dict(t) -> dict:
    return {
        "id": t.id,
        "op": t.op,
        "impl": getattr(t, "impl", None),
        "inputs": t.in_,
        "kwargs": getattr(t, "kwargs", {}) or {},
        "output": t.out,
        "device": getattr(t, "device", "cpu"),
        "cast_input0_to_complex": bool(getattr(t, "cast_input0_to_complex", False)),
    }


def extract_test_specs(cfg: Config) -> List[dict]:
    """Convert a validated `Config` into a normalized list of plain-Python test spec dictionaries.

    This flattens `cfg.tests` into serializable dicts used by runners/APIs. For normal tests, it emits a
    single dict with keys like `id`, `op`, `inputs`, `kwargs`, `output`, `device`, and `cast_input0_to_complex`.
    For `PairTest`, it emits a dict of the form:
        {"type": "pair", id: base_id, "a": <testcase_dict>, "b": <testcase_dict>}

    Args:
        cfg: Parsed and validated test configuration containing `tests` entries (TestCase or PairTest).

    Returns:
        A list of dictionaries, one per test item in `cfg.tests`, where PairTest items are represented
        as a single "pair" dict containing two testcase dicts.
    """

    specs: List[dict] = []
    for item in cfg.tests:
        if isinstance(item, PairTest):
            specs.append({
                "type": "pair",
                "id": item.id,
                "a": _testcase_to_dict(item.a),
                "b": _testcase_to_dict(item.b),
            })
        else:
            specs.append(_testcase_to_dict(item))
    return specs

def build_input_specs(cfg: Config, test: dict) -> List[Spec]:
    """Build Specs for positional inputs in `test["inputs"]`,
    resolving presets and symbolic dims from `cfg`."""
    resolved_inputs = [resolve_node(n, cfg.presets) for n in test["inputs"]]
    return [node_to_spec(n, cfg.dims) for n in resolved_inputs]

def build_kw_specs(cfg: Config, test: dict) -> Dict[str, Spec]:
    """"Build Specs for keyword arguments defined by `test["kwargs"]`, a mapping `{name: node}` 
    describing how to generate each kwarg value (resolved via cfg presets/dims)."""
    raw_kwargs = test.get("kwargs") or {}
    resolved_kwargs = {k: resolve_node(v, cfg.presets) for k, v in raw_kwargs.items()}
    return {k: node_to_spec(v, cfg.dims) for k, v in resolved_kwargs.items()}

def build_specs(cfg: Config, test: dict) -> Tuple[List[Spec], Dict[str, Spec]]:
    """Build sampling specs for one test case by resolving presets and symbolic dims.

    This is a convenience wrapper that returns both:
    - positional input specs (in the same order as `test["inputs"]`), and
    - keyword argument specs (a dict for `test.get("kwargs", {})`).

    Resolution steps:
    1) Expand any `RefNode` using `cfg.presets` (preset substitution).
    2) Convert node structures into internal `Spec` objects via `node_to_spec`, resolving symbolic
        shape/range entries using `cfg.dims`.

    Args:
        cfg: Validated Config providing:
            - dims: mapping of symbolic dimension names to ints
            - presets: mapping of preset names to input nodes
        test: Normalized test dict containing:
            - "inputs": list of input nodes
            - optional "kwargs": mapping of kw names to input nodes

    Returns:
        (input_specs, kw_specs)
        - input_specs: List[Spec] aligned with `test["inputs"]`
        - kw_specs: Dict[str, Spec] aligned with `test.get("kwargs", {})`
    """
    return build_input_specs(cfg, test), build_kw_specs(cfg, test)

# -------------------------
# kwargs export helpers
# -------------------------

def split_export_args(values, specs, kwargs_values=None, kwargs_specs=None):
    """Split export-adapted args/kwargs into tensor inputs vs constant inputs for backend export.

    This partitions positional arguments into:
    - `tensor_args`: runtime-provided tensors (plus any kw-only tensor values appended at the end)
    - `const_args`: non-tensor constants and TensorListMarker placeholders
    - `arg_is_tensor`: boolean mask describing whether each original positional arg position is
        supplied by `tensor_args` (True) or `const_args` (False)

    Keyword arguments are split similarly:
    - `const_kwargs`: kw entries that are not tensors
    - `kw_tensor_keys`: ordered list of kw names whose tensor values are appended to `tensor_args`

    Args:
        values: Export-ready positional values (after `to_export_values`), aligned with `specs`.
        specs: Positional Specs describing which values are tensors vs constants (and list markers).
        kwargs_values: Optional export-ready kw values (after `to_export_kwargs`).
        kwargs_specs: Optional kw Specs describing kwarg structure.

    Returns:
        (tensor_args, const_args, arg_is_tensor, const_kwargs, kw_tensor_keys)
    """
    tensor_args: List[Any] = []
    const_args: List[Any] = []
    arg_is_tensor: List[bool] = []

    for v, s in zip(values, specs):
        if isinstance(s, TensorSpec) and (not isinstance(s, ConstTensorSpec)):
            arg_is_tensor.append(True)
            tensor_args.append(v)
            continue

        if isinstance(s, ListSpec) and isinstance(s.elem, TensorSpec) and (not isinstance(s.elem, ConstTensorSpec)):
            if not isinstance(v, list) or len(v) == 0 or not all(torch.is_tensor(t) for t in v):
                raise TypeError("Expected non-empty list[Tensor] for ListSpec(elem=TensorSpec)")
            arg_is_tensor.append(False)
            const_args.append(TensorListMarker(n=len(v)))
            tensor_args.extend(v)
            continue

        arg_is_tensor.append(False)
        const_args.append(v)

    const_kwargs: Dict[str, Any] = {}
    kw_tensor_keys: List[str] = []

    if kwargs_specs:
        kwargs_values = kwargs_values or {}
        for k, s in kwargs_specs.items():
            if k not in kwargs_values:
                raise KeyError(f"Missing kwargs value for '{k}'")
            v = kwargs_values[k]

            if isinstance(s, TensorSpec) and (not isinstance(s, ConstTensorSpec)):
                tensor_args.append(v)
                kw_tensor_keys.append(k)
                continue

            if isinstance(s, ListSpec) and isinstance(s.elem, TensorSpec) and (not isinstance(s.elem, ConstTensorSpec)):
                raise TypeError("List[Tensor] kwargs are not supported")

            const_kwargs[k] = v

    return tensor_args, const_args, arg_is_tensor, const_kwargs, kw_tensor_keys