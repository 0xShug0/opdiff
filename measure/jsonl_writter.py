"""JSONL writer for test results.

This module provides `JSONLWriter`, a small utility that writes run metadata and per-test
results as JSON Lines (one JSON object per line). The first record for each run is a
"meta" entry capturing environment versions and user-provided run parameters; subsequent
records are sanitized "item" entries containing statuses, timings, and diffs.
"""

from __future__ import annotations
import sys
import datetime as _dt
import json
import os
from typing import Any, Dict, Optional


def _get_versions() -> dict[str, str | None]:
    """Collect best-effort versions for key runtime packages and torch runtime capabilities."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    def _pkg(name: str) -> str | None:
        """Return installed distribution version for `name`, or None if unavailable."""
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return None
        except Exception:
            return None

    def _mod(name: str) -> str | None:
        """Return imported module __version__ for `name`, or None if unavailable."""
        try:
            m = __import__(name)
            return getattr(m, "__version__", None)
        except Exception:
            return None

    d: dict[str, str | None] = {
        "python": sys.version,
        "torch": _pkg("torch") or _mod("torch"),
        "torchvision": _pkg("torchvision") or _mod("torchvision"),
        "torchaudio": _pkg("torchaudio") or _mod("torchaudio"),
        "coremltools": _pkg("coremltools") or _mod("coremltools"),
        "onnxruntime": _pkg("onnxruntime") or _mod("onnxruntime"),
        "executorch": _pkg("executorch") or _mod("executorch"),
    }

    # Torch runtime bits are useful for debugging backend differences; gather best-effort.
    try:
        import torch

        d["torch_cuda"] = getattr(torch.version, "cuda", None)
        d["torch_mps_available"] = str(
            bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )
    except Exception:
        d["torch_cuda"] = None
        d["torch_mps_available"] = None

    return d


class JSONLWriter:
    """Append-only JSONL logging for run metadata and per-item results."""
    def __init__(
        self,
        path: str,
        *,
        format_name: str = "opdiff_run",
        format_version: int = 1,
        fsync: bool = True,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a writer for a JSONL file.

        Args:
            path: Output JSONL path (must be non-empty).
            format_name: Logical name for the record schema.
            format_version: Schema version number for consumers.
            fsync: If True, fsync after each record for durability.
            extra_meta: Optional extra metadata merged into the run "meta" record.
        """
        if not path:
            raise ValueError("output_path is required (non-empty)")
        self.path = path
        self.format_name = format_name
        self.format_version = int(format_version)
        self.fsync = bool(fsync)
        self.extra_meta = extra_meta or {}

    def build_meta(self, *, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build the "meta" record payload for a run."""
        extra = dict(self.extra_meta)
        extra.setdefault("yaml", run_info.get("yaml"))
        extra.setdefault("params", run_info.get("params"))
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "env": _get_versions(),
            "extra": extra,
        }

    def start_run(self, *, run_info: Dict[str, Any], mode: str = "w") -> None:
        """Initialize the output file and write the run "meta" record."""
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a' (got {mode!r})")

        # Ensure parent directory exists so run initialization is robust.
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # In overwrite mode, truncate the file before writing the new run header.
        if mode == "w":
            open(self.path, "w").close()

        # Always write a run header so multiple runs can be appended to one file.
        meta = self.build_meta(run_info=run_info)
        self.append({"type": "meta", "meta": meta})
        
    def append(self, rec: Dict[str, Any]) -> None:
        """Append a single JSON record as one line."""
        # Intentionally avoid json default hooks so non-serializable objects fail loudly.
        line = json.dumps(rec)
        with open(self.path, "a", encoding="utf-8") as jf:
            jf.write(line + "\n")
            jf.flush()
            if self.fsync:
                os.fsync(jf.fileno())

    def sanitize_output(self, out_item: Dict[str, Any]) -> Dict[str, Any]:
        """Project an internal per-item result into a stable JSONL-friendly schema."""
        item = {
            "type": "item",
            "id": out_item.get("id"),
            "test_type": out_item.get("type"),
            "cases": out_item.get("cases", []),
            "baseline": None,
            "backends": [],
        }

        # Baseline is optional and only present for single-test items when enabled.
        if out_item.get("baseline") is not None:
            b = dict(out_item["baseline"])
            item["baseline"] = b
        
        # Keep only fields intended for persistence; ignore incidental/internal keys.
        for be in out_item.get("backends", []):
            be2: Dict[str, Any] = {
                "backend": be.get("backend"),
                "res": [],
                "times_all": be.get("times_all", None),
                "diff_all": be.get("diff_all", None),
            }

            for s in be.get("res", []):
                s2 = dict(s)
                be2["res"].append(s2)

            item["backends"].append(be2)

        return item

    def write(self, out_item: Dict[str, Any]) -> None:
        """Sanitize and append one per-item result record."""
        self.append(self.sanitize_output(out_item))