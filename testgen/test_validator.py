"""
Pydantic v2 schema and validation for YAML op test configurations.

This module defines:
- Node models for declarative inputs (TensorNode, ScalarNode, ListNode, TupleNode,
  OptionalNode, ConstNode, ConstTensorNode, ScalarTensorNode) plus `RefNode` for
  preset references.
- Op specifications for runtime construction (`ModuleNode`, `ConstructNode`) and
  template variants (`TemplateModuleNode`, `TemplateComparePairNode`).
- Test containers (`TestCase`, `PairTest`) and the top-level `Config` (dims, presets, tests).

Validation utilities enforce supported dtype/kind/init values, non-empty shapes,
symbol-or-number fields (dims/ranges), and helpful typo detection for unknown
`TestCase` fields.

`format_validation_error()` turns Pydantic errors into concise, test-scoped
messages that include the test id/op when available.
"""

from __future__ import annotations
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union, cast
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, ConfigDict
from pydantic.types import JsonValue
import difflib

Dim = Union[int, str]
Rnge = Union[int, float, str]  # numeric or symbolic string

DTYPE_SET: Set[str] = {
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64", "uint8",
    "bool",
    "complex64", "complex128",
}
KIND_SET: Set[str] = {"float", "int", "bool"}
INIT_SET: Set[str] = {"normal", "uniform", "randint", "zeros", "ones", "bernoulli"}


# -------------------------
# Shared validation helpers
# -------------------------
def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def _validate_symbol_or_number(
    v: Any, *, where: str, allow_float: bool = True, coerce_int: bool = False
) -> Union[int, float, str]:
    if _is_nonempty_str(v):
        return cast(str, v).strip()
    if isinstance(v, bool):
        raise ValueError(f"{where} must be a number or a non-empty symbolic string")
    if isinstance(v, int):
        return int(v) if coerce_int else v
    if allow_float and isinstance(v, float):
        return int(v) if coerce_int else v
    raise ValueError(f"{where} must be a number or a non-empty symbolic string")

def _validate_shape(v: List[Dim]) -> List[Dim]:
    if not v:
        raise ValueError("shape cannot be empty")
    for d in v:
        if isinstance(d, int):
            if d < 0:
                raise ValueError("shape dims must be >= 0")
        elif not _is_nonempty_str(d):
            raise ValueError("symbolic dims must be non-empty strings")
    return v

def _validate_kind(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    if v not in KIND_SET:
        raise ValueError(f"unsupported kind '{v}'")
    return v

def _validate_dtype(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    if v not in DTYPE_SET:
        raise ValueError(f"unsupported dtype '{v}'")
    return v

def _validate_init(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    if v not in INIT_SET:
        raise ValueError(f"unsupported init '{v}'")
    return v


# --------------------------------
# constructor / module specifications
# --------------------------------
class ConstructNode(BaseModel):
    type: Literal["construct"]
    path: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_nested_nodes_in_args(cls, v: Any) -> Any:
        # Allow nested {"type": "construct"/"module", ...} inside construct args
        if isinstance(v, list):
            return [parse_node(x) for x in v]
        return v

    @field_validator("kwargs", mode="before")
    @classmethod
    def _coerce_nested_nodes_in_kwargs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: parse_node(x) for k, x in v.items()}
        return v

class ModuleNode(BaseModel):
    type: Literal["module"]
    path: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_nodes_in_args(cls, v: Any) -> Any:
        # Force dict nodes to become models, instead of falling into JsonValue
        if isinstance(v, list):
            return [parse_node(x) for x in v]
        return v

    @field_validator("kwargs", mode="before")
    @classmethod
    def _coerce_nodes_in_kwargs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: parse_node(x) for k, x in v.items()}
        return v

class VarNode(BaseModel):
    type: Literal["var"]
    name: str

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("var.name must be a non-empty string")
        return v.strip()

class TemplateModuleNode(BaseModel):
    type: Literal["template_module"]
    path: str

    # independent vars (cartesian product)
    vars: Dict[str, List[JsonValue]] = Field(default_factory=dict)

    # coupled vars (each entry is a bundle of assignments)
    cases: List[Dict[str, JsonValue]] = Field(default_factory=list)

    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("vars")
    @classmethod
    def _vars_ok(cls, v: Dict[str, List[JsonValue]]) -> Dict[str, List[JsonValue]]:
        if not isinstance(v, dict):
            raise ValueError("template_module.vars must be a dict")
        for k, lst in v.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("template_module.vars keys must be non-empty strings")
            if not isinstance(lst, list) or len(lst) == 0:
                raise ValueError(f"template_module.vars['{k}'] must be a non-empty list")
        return v

    @field_validator("cases")
    @classmethod
    def _cases_ok(cls, v: List[Dict[str, JsonValue]]) -> List[Dict[str, JsonValue]]:
        if not isinstance(v, list):
            raise ValueError("template_module.cases must be a list")
        for i, d in enumerate(v):
            if not isinstance(d, dict) or len(d) == 0:
                raise ValueError(f"template_module.cases[{i}] must be a non-empty dict")
            for k in d.keys():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"template_module.cases[{i}] keys must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _cases_keys_must_be_in_vars(self) -> "TemplateModuleNode":
        # If cases are provided, require their keys to be declared in vars
        if self.cases:
            declared = set(self.vars.keys())
            for i, d in enumerate(self.cases):
                extra = [k for k in d.keys() if k not in declared]
                if extra:
                    raise ValueError(
                        f"template_module.cases[{i}] has keys not declared in vars: {extra}"
                    )
        return self

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_nodes_in_args(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [parse_node(x) for x in v]
        return v

    @field_validator("kwargs", mode="before")
    @classmethod
    def _coerce_nodes_in_kwargs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: parse_node(x) for k, x in v.items()}
        return v


class CompareSideNode(BaseModel):
    impl: str
    type: Literal["module"] = "module"
    path: str
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_nodes_in_args(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [parse_node(x) for x in v]
        return v

    @field_validator("kwargs", mode="before")
    @classmethod
    def _coerce_nodes_in_kwargs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: parse_node(x) for k, x in v.items()}
        return v


class TemplateComparePairNode(BaseModel):
    type: Literal["template_compare_pair"]

    vars: Dict[str, List[JsonValue]] = Field(default_factory=dict)
    cases: List[Dict[str, JsonValue]] = Field(default_factory=list)

    # shared config to dedupe
    common: Dict[str, Any] = Field(default_factory=dict)

    a: CompareSideNode
    b: CompareSideNode

    @field_validator("vars")
    @classmethod
    def _vars_ok(cls, v: Dict[str, List[JsonValue]]) -> Dict[str, List[JsonValue]]:
        if not isinstance(v, dict):
            raise ValueError("template_compare_pair.vars must be a dict")
        for k, lst in v.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("template_compare_pair.vars keys must be non-empty strings")
            if not isinstance(lst, list) or len(lst) == 0:
                raise ValueError(f"template_compare_pair.vars['{k}'] must be a non-empty list")
        return v

    @field_validator("cases")
    @classmethod
    def _cases_ok(cls, v: List[Dict[str, JsonValue]]) -> List[Dict[str, JsonValue]]:
        if not isinstance(v, list):
            raise ValueError("template_compare_pair.cases must be a list")
        for i, d in enumerate(v):
            if not isinstance(d, dict) or len(d) == 0:
                raise ValueError(f"template_compare_pair.cases[{i}] must be a non-empty dict")
            for k in d.keys():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"template_compare_pair.cases[{i}] keys must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _cases_keys_must_be_in_vars(self) -> "TemplateComparePairNode":
        if self.cases:
            declared = set(self.vars.keys())
            for i, d in enumerate(self.cases):
                extra = [k for k in d.keys() if k not in declared]
                if extra:
                    raise ValueError(
                        f"template_compare_pair.cases[{i}] has keys not declared in vars: {extra}"
                    )
        return self

def parse_node(x: Any) -> Any:
    if isinstance(x, dict):
        t = x.get("type")
        if t == "construct":
            return ConstructNode.model_validate(x)
        if t == "module":
            return ModuleNode.model_validate(x)
        if t == "template_module":
            return TemplateModuleNode.model_validate(x)
        if t == "template_compare_pair":
            return TemplateComparePairNode.model_validate(x)
        if t == "var":
            return VarNode.model_validate(x)
    return x

# ----------------
# Shared mixins
# ----------------
class KindDtypeMixin(BaseModel):
    kind: Optional[str] = None
    dtype: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def _kind_ok(cls, v: Optional[str]) -> Optional[str]:
        return _validate_kind(v)

    @field_validator("dtype")
    @classmethod
    def _dtype_ok(cls, v: Optional[str]) -> Optional[str]:
        return _validate_dtype(v)

    @model_validator(mode="after")
    def _kind_or_dtype_required(self) -> "KindDtypeMixin":
        if self.kind is None and self.dtype is None:
            raise ValueError("must specify either 'dtype' or 'kind'")
        return self


# ------------------------
# input node algebra
# ------------------------
class RefNode(BaseModel):
    type: Literal["ref"]
    ref: Union[str, VarNode]

    @field_validator("ref")
    @classmethod
    def _ref_must_exist_in_presets(cls, v, info):
        preset_keys = None if info.context is None else info.context.get("preset_keys")
        if preset_keys is None:
            return v
        if isinstance(v, str) and v not in preset_keys:
            raise ValueError(f"unknown preset ref '{v}'")
        return v

class TensorNode(KindDtypeMixin):
    type: Literal["tensor"]

    shape: List[Dim]
    init: Optional[str] = None
    low: Optional[Rnge] = 0
    high: Optional[Rnge] = 10
    mean: float = 0.0
    std: float = 1.0
    p: float = 0.5
    requires_grad: bool = False

    @field_validator("init")
    @classmethod
    def _init_ok(cls, v: Optional[str]) -> Optional[str]:
        return _validate_init(v)

    @field_validator("shape")
    @classmethod
    def _shape_ok(cls, v: List[Dim]) -> List[Dim]:
        return _validate_shape(v)

    @field_validator("low", "high")
    @classmethod
    def _low_high_ok(cls, v: Optional[Rnge], info) -> Optional[Rnge]:
        if v is None:
            return None
        return _validate_symbol_or_number(v, where=f"tensor.{info.field_name}")  # type: ignore[return-value]


class ScalarNode(KindDtypeMixin):
    type: Literal["scalar"]

    value: Optional[Union[int, float, bool]] = None
    low: Optional[Rnge] = None
    high: Optional[Rnge] = None
    p: Optional[float] = None  # for bool sampling

    @model_validator(mode="after")
    def _range_or_value_ok(self) -> "ScalarNode":
        if self.value is not None:
            return self
        if (self.low is None) != (self.high is None):
            raise ValueError("scalar must specify both low and high (or neither)")

        if self.kind == "bool":
            if self.p is not None and not (0.0 <= self.p <= 1.0):
                raise ValueError("scalar.p must be in [0,1]")
            return self

        if isinstance(self.low, str) or isinstance(self.high, str):
            if isinstance(self.low, str) and not _is_nonempty_str(self.low):
                raise ValueError("scalar.low symbolic value must be a non-empty string")
            if isinstance(self.high, str) and not _is_nonempty_str(self.high):
                raise ValueError("scalar.high symbolic value must be a non-empty string")
            return self

        if self.low is not None and self.high is not None:
            if not (cast(Union[int, float], self.low) < cast(Union[int, float], self.high)):
                raise ValueError("scalar.low must be < scalar.high")
        return self


class ScalarTensorNode(KindDtypeMixin):
    type: Literal["scalar_tensor"]

    value: Optional[Union[int, float, bool]] = None
    init: Optional[str] = None
    low: Optional[Rnge] = 0
    high: Optional[Rnge] = 10
    mean: float = 0.0
    std: float = 1.0
    p: float = 0.5

    @field_validator("init")
    @classmethod
    def _init_ok(cls, v: Optional[str]) -> Optional[str]:
        return _validate_init(v)

    @field_validator("low", "high")
    @classmethod
    def _low_high_ok(cls, v: Optional[Rnge], info) -> Optional[Rnge]:
        if v is None:
            return None
        return _validate_symbol_or_number(v, where=f"scalar_tensor.{info.field_name}")  # type: ignore[return-value]


class IntListNode(BaseModel):
    type: Literal["int_list"]
    elems: List[int]

    @field_validator("elems")
    @classmethod
    def _elems_ok(cls, v: List[int]) -> List[int]:
        if v:
            return v
        raise ValueError("int_list.elems cannot be empty")


class ListNode(BaseModel):
    type: Literal["list"]
    len: Rnge
    elem: "Node"

    @field_validator("len")
    @classmethod
    def _len_ok(cls, v: Rnge) -> Rnge:
        v2 = _validate_symbol_or_number(v, where="list.len", allow_float=True, coerce_int=True)
        if isinstance(v2, int) and v2 < 0:
            raise ValueError("list.len must be >= 0")
        return cast(Rnge, v2)


class TupleNode(BaseModel):
    type: Literal["tuple"]
    elems: List["Node"]


class OptionalNode(BaseModel):
    type: Literal["optional"]
    p_none: float = 0.0
    elem: "Node"

    @field_validator("p_none")
    @classmethod
    def _p_ok(cls, v: float) -> float:
        if 0.0 <= v <= 1.0:
            return v
        raise ValueError("p_none must be in [0,1]")


class ConstNode(BaseModel):
    type: Literal["const"]
    value: JsonValue


class ConstTensorNode(KindDtypeMixin):
    type: Literal["const_tensor"]

    shape: List[Dim]
    init: Optional[str] = None
    low: Optional[Rnge] = 0
    high: Optional[Rnge] = 10
    mean: float = 0.0
    std: float = 1.0
    p: float = 0.5
    value: Optional[JsonValue] = None

    @field_validator("init")
    @classmethod
    def _init_ok(cls, v: Optional[str]) -> Optional[str]:
        return _validate_init(v)

    @field_validator("shape")
    @classmethod
    def _shape_ok(cls, v: List[Dim]) -> List[Dim]:
        return _validate_shape(v)

    @field_validator("low", "high")
    @classmethod
    def _low_high_ok(cls, v: Optional[Rnge], info) -> Optional[Rnge]:
        if v is None:
            return None
        return _validate_symbol_or_number(v, where=f"const_tensor.{info.field_name}")  # type: ignore[return-value]


Node = Annotated[
    Union[
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
    ],
    Field(discriminator="type"),
]

ListNode.model_rebuild()
TupleNode.model_rebuild()
OptionalNode.model_rebuild()
ModuleNode.model_rebuild()
TemplateModuleNode.model_rebuild()
TemplateComparePairNode.model_rebuild()
# ------------------------
# test case / config
# ------------------------
class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Optional[str] = None
    impl: Optional[str] = None
    op: Union[str, ModuleNode, TemplateModuleNode, TemplateComparePairNode]
    in_: List[Node] = Field(alias="in")
    kwargs: Dict[str, Node] = Field(default_factory=dict)
    out: Optional[Node] = None
    device: Literal["cpu", "gpu", "cuda", "mps"] = "cpu"
    cast_input0_to_complex: bool = False

    @model_validator(mode="before")
    @classmethod
    def _typo_check_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        allowed = set(cls.model_fields.keys()) | {"in"}  # include alias
        extra = [k for k in data.keys() if k not in allowed]
        if not extra:
            return data

        msgs = []
        for k in extra:
            candidates = difflib.get_close_matches(k, sorted(allowed), n=1, cutoff=0.8)
            if candidates:
                msgs.append(f"unknown field '{k}' (did you mean '{candidates[0]}'?)")
            else:
                msgs.append(f"unknown field '{k}'")

        raise ValueError("; ".join(msgs))

class PairTest(BaseModel):
    type: Literal["pair"] = "pair"
    id: Optional[str] = None
    a: TestCase
    b: TestCase

TestItem = Union[TestCase, PairTest]

class Config(BaseModel):
    dims: Dict[str, int] = Field(default_factory=dict)
    presets: Dict[str, Node] = Field(default_factory=dict)
    tests: List[TestItem]

    @model_validator(mode="after")
    def _validate_dims(self) -> "Config":
        for k, v in self.dims.items():
            if not isinstance(v, int):
                raise ValueError(f"dims.{k} must be int")
            if v < 0:
                raise ValueError(f"dims.{k} must be >= 0")
        return self


# ------------------------
# Error formatting (id + op)
# ------------------------
def _op_display(op_field: Any) -> str:
    if isinstance(op_field, str):
        return op_field
    if isinstance(op_field, dict):
        t = op_field.get("type")
        p = op_field.get("path")
        if t and p:
            return f"{t}:{p}"
    if isinstance(op_field, ModuleNode):
        return f"{op_field.type}:{op_field.path}"
    return repr(op_field)


def _test_label_from_raw_tests(raw_tests: Any, test_idx: Optional[int]) -> str:
    if test_idx is None or not isinstance(raw_tests, list) or not (0 <= test_idx < len(raw_tests)):
        return "test=unknown"

    t = raw_tests[test_idx]
    if isinstance(t, dict):
        tid = t.get("id")
        op = _op_display(t.get("op"))
        
        if t.get("type") == "pair":
            return f"id={tid or 'pair'} op=pair"
        
        if tid and op:
            return f"id={tid} op={op}"
        if tid:
            return f"id={tid}"
        if op:
            return f"op={op}"
        return f"index={test_idx}"

    return f"index={test_idx}"


def _summarize_root_input(inp: Any) -> str:
    if not isinstance(inp, dict):
        return ""
    dims = inp.get("dims", {})
    presets = inp.get("presets", {})
    tests = inp.get("tests", [])
    if isinstance(dims, dict) and isinstance(presets, dict) and isinstance(tests, list):
        return f" | input_summary=dims:{len(dims)} presets:{len(presets)} tests:{len(tests)}"
    return ""


def format_validation_error(e: ValidationError, raw_data: Optional[dict] = None) -> str:
    raw_tests = (raw_data or {}).get("tests", [])
    lines: List[str] = []

    for err in e.errors():
        loc = err.get("loc", ())
        msg = err.get("msg", "") or ""
        inp = err.get("input", None)
        typ = err.get("type", "") or ""
        ctx = err.get("ctx") or {}

        # If ctx carries a more specific message, surface it, but avoid duplication.
        if "error" in ctx:
            err_txt = str(ctx["error"])
            if not msg:
                msg = err_txt
            else:
                # Pydantic v2 often formats msg like "Value error, <detail>"
                # and ctx["error"] repeats <detail>. Don't print it twice.
                if err_txt not in msg:
                    if msg.startswith("Value error"):
                        msg = f"{msg}: {err_txt}"
                    else:
                        msg = f"{msg} | {err_txt}"

        test_idx: Optional[int] = None
        if len(loc) >= 2 and loc[0] == "tests" and isinstance(loc[1], int):
            test_idx = loc[1]

        label = _test_label_from_raw_tests(raw_tests, test_idx)
        where = ".".join(str(x) for x in loc) if loc else "<root>"
        suffix = f" (type={typ})" if typ else ""

        if where == "<root>":
            lines.append(f"[{label}] {where}: {msg}{suffix}{_summarize_root_input(inp)}")
            continue

        inp_repr = repr(inp)
        if len(inp_repr) > 180:
            inp_repr = inp_repr[:177] + "..."
        lines.append(f"[{label}] {where}: {msg}{suffix} | input={inp_repr}")

    return "\n".join(lines)

