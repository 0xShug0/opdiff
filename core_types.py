import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
# -------------------------
# dtype mapping
# -------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

DEFAULT_DTYPE_BY_KIND = {
    "float": torch.float32,
    "int": torch.int64,
    "bool": torch.bool,
}

# CoreMLTools input signature dtype mapping
TORCH_TO_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

@dataclass(frozen=True)
class Precision:
    float: torch.dtype = torch.float32
    int: torch.dtype = torch.int64
    bool: torch.dtype = torch.bool
    fp16: torch.dtype = torch.float16


@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    kind: str = "float"  # "float" | "int" | "bool"
    dtype: Optional[torch.dtype] = None
    device: Union[str, torch.device] = "cpu"
    dist: Optional[str] = None

    low: int = 0
    high: int = 10
    mean: float = 0.0
    std: float = 1.0
    p: float = 0.5
    requires_grad: bool = False


@dataclass(frozen=True)
class ScalarSpec:
    kind: str  # "int" | "float" | "bool"
    low: float = -3.0
    high: float = 3.0
    choices: Optional[Tuple[Any, ...]] = None
    dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class ScalarTensorSpec:
    kind: str = "float"  # "float" | "int" | "bool"
    dtype: Optional[torch.dtype] = None
    device: Union[str, torch.device] = "cpu"
    dist: Optional[str] = None

    low: int = 0
    high: int = 10
    mean: float = 0.0
    std: float = 1.0
    p: float = 0.5
    value: Any = None


@dataclass(frozen=True)
class ListSpec:
    elem: Any
    length: int


@dataclass(frozen=True)
class OptionalSpec:
    elem: Any
    none_prob: float = 0.0


@dataclass(frozen=True)
class ConstSpec:
    value: Any


@dataclass(frozen=True)
class ConstTensorSpec:
    tensor: TensorSpec
    value: Any = None


@dataclass(frozen=True)
class TensorListMarker:
    n: int
    
Spec = Union[TensorSpec, ScalarSpec, ScalarTensorSpec, ListSpec, OptionalSpec, ConstSpec, ConstTensorSpec]





