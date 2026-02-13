#!/usr/bin/env python3
"""
Example CLI commands (repo_root/examples/demo.yaml)

# 0) Show available backend preset names (one per line)
python cli.py list-presets

# 1) Validate a YAML suite (loads + expands tests; fails if YAML invalid or selected ids missing)
python cli.py validate-yaml examples/demo.yaml

# 2) List expanded test ids (after template expansion / cases / pair expansion)
python cli.py list-tests examples/demo.yaml

# 2b) List only ids matching patterns (useful to discover exact expanded ids)
python cli.py list-tests examples/demo.yaml --test-ids 'mod_linear_*;toy_ln_*;fft_*'

# 3) Run a small subset on a single backend (no baseline diff; writes JSONL)
python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;op_add_with_alpha_kw;op_mul_scalar_tensor' \
  --out runs/demo_out.jsonl

# 4) Run multiple backends and compare each vs the baseline backend (single-item tests only)
python cli.py run examples/demo.yaml \
  --backends 'torch_cpu_fp32;onnx_cpu_fp32' \
  --baseline torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;op_where_bool_cond;op_cat_tensor_list' \
  --out runs/demo_out.jsonl

# 5) Run module tests (torch.nn.* module specs) on multiple backends
python cli.py run examples/demo.yaml \
  --backends 'torch_cpu_fp32;torch_cpu_fp16' \
  --test-ids 'mod_linear_forward;mod_layernorm_impl_cpu;mod_sequential_with_construct' \
  --out runs/demo_out.jsonl

# 6) Run templated tests (template_module expands into multiple concrete tests)
# - templated tests must use prefix matching 
python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids 'mod_linear_template*' \
  --out runs/demo_out.jsonl

# 7) Run templated tests with cases constraints (template_module + cases restrict cartesian product)
python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids 'mod_linear_cases_match_input_features*'\
  --out runs/demo_out.jsonl

# 8) Run a pair test (template_compare_pair): compares A vs B within the SAME backends
python cli.py run examples/demo.yaml \
  --backends 'torch_cpu_fp32;onnx_cpu_fp32' \
  --test-ids toy_ln_vs_rms_pair \
  --out runs/demo_out.jsonl

# 9) Skip specific (id, backend) combinations
# - format: id:backend;id:backend;...  (backend can be "*")
python cli.py run examples/demo.yaml \
  --backends 'torch_cpu_fp32;onnx_cpu_fp32' \
  --test-ids 'op_add_ref_ref;op_gather_dim1;op_permute_int_list' \
  --skips 'op_gather_dim1:onnx_cpu_fp32' \
  --out runs/demo_out.jsonl

# 10) Append to an existing JSONL (writes a new run header + item records)
python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids op_add_ref_ref \
  --out runs/demo_out.jsonl

python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids op_add_with_alpha_kw \
  --append \
  --out runs/demo_out.jsonl

# 11) Mode 2: record per-repeat timings/outputs and per-repeat diffs (more detailed JSONL)
python cli.py run examples/demo.yaml \
  --backends 'onnx_cpu_fp32' \
  --baseline torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;mod_sequential_with_construct' \
  --mode 2 \
  --repeats 10 \
  --num-sampled-inputs 3 \
  --out runs/demo_out.jsonl

# 12) Timed isolation: setting --timeout-s automatically selects the timed subprocess runner
python cli.py run examples/demo.yaml \
  --backends 'onnx_cpu_fp32' \
  --baseline torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;mod_sequential_with_construct' \
  --mode 2 \
  --repeats 10 \
  --num-sampled-inputs 3 \
  --timeout-s 10 \
  --out runs/demo_out.jsonl

# 13) Enforce non-zero outputs (fail fast if first output is all zeros)
python cli.py run examples/demo.yaml \
  --backends torch_cpu_fp32 \
  --test-ids op_add_ref_ref \
  --no-allow-all-zero-output \
  --out runs/demo_out.jsonl

# 14) Backend overrides via file (recommended for anything non-trivial)
python cli.py run examples/demo.yaml \
  --backends 'onnx_cpu_fp32;torch_cpu_fp16' \
  --baseline torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;op_cat_tensor_list' \
  --backend-overrides-file examples/backend_overrides.json \
  --out runs/demo_out.jsonl
  
# 14b) Overrides via cmd
python cli.py run examples/demo.yaml \
  --backends 'onnx_cpu_fp32;torch_cpu_fp32' \
  --baseline torch_cpu_fp32 \
  --test-ids 'op_add_ref_ref;op_cat_tensor_list' \
  --backend-overrides-json '{"onnx_cpu_fp32":{"onnx_overrides":{"opset":19}}}' \
  --out runs/demo_out.jsonl

# 15) Disable cache clean
python cli.py run examples/demo.yaml \
  --backends onnx_cpu_fp32 \
  --test-ids op_add_ref_ref \
  --no-clean-tmp \
  --out runs/demo_out.jsonl
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_import_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _split_semis(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [p.strip() for p in str(s).split(";")]
    parts = [p for p in parts if p]
    return parts if parts else None


def _parse_skips(s: Optional[str]) -> Optional[List[Dict[str, str]]]:
    toks = _split_semis(s)
    if not toks:
        return None
    out: List[Dict[str, str]] = []
    for t in toks:
        if ":" not in t:
            raise ValueError(f"Bad --skips token (expected id:backend): {t!r}")
        tid, backend = t.split(":", 1)
        tid = tid.strip()
        backend = backend.strip()
        if not tid or not backend:
            raise ValueError(f"Bad --skips token (empty id/backend): {t!r}")
        out.append({"id": tid, "backend": backend})
    return out if out else None


def _load_overrides(path: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path:
        return None
    p = Path(path)
    data: Any
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            raise RuntimeError("YAML overrides requested but PyYAML is not installed")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("backend overrides must be a dict: {preset_name: {override_group: {...}}}")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            raise ValueError("backend overrides must be a dict[str, dict]")
        out[k] = v
    return out


def _load_overrides_json(s: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not s:
        return None
    data: Any = json.loads(s)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("backend overrides must be a dict: {preset_name: {override_group: {...}}}")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            raise ValueError("backend overrides must be a dict[str, dict]")
        out[k] = v
    return out

def _cmd_list_presets() -> int:
    _ensure_import_path()
    from pipeline.presets import BACKEND_PRESETS

    for name in sorted(BACKEND_PRESETS.keys()):
        print(name)
    return 0


def _cmd_list_tests(args: argparse.Namespace) -> int:
    _ensure_import_path()
    from pipeline.core import resolve_tests

    test_ids = _split_semis(args.test_ids)
    _cfg, tests = resolve_tests(args.yaml_path, test_ids=test_ids)
    for t in tests:
        print(str(t.get("id", "")))
    return 0


def _cmd_validate_yaml(args: argparse.Namespace) -> int:
    _ensure_import_path()
    from pipeline.core import resolve_tests

    test_ids = _split_semis(args.test_ids)
    resolve_tests(args.yaml_path, test_ids=test_ids)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _ensure_import_path()
    from pipeline.api import run as pipeline_run

    backends = _split_semis(args.backends) or []
    if not backends:
        raise ValueError("--backends is required and must contain at least one preset name")

    test_ids = _split_semis(args.test_ids)
    skipped_tests = _parse_skips(args.skips)
    backend_overrides = _load_overrides(args.backend_overrides_file) or _load_overrides_json(args.backend_overrides_json)

    output_mode = "a" if args.append else "w"
    timeout_s = float(args.timeout_s) if args.timeout_s is not None else None

    pipeline_run(
        yaml_path=args.yaml_path,
        target_backends=backends,
        test_ids=test_ids,
        skipped_tests=skipped_tests,
        baseline_backend=args.baseline,
        adhoc_pairs=None,
        repeats=int(args.repeats),
        num_sampled_inputs=int(args.num_sampled_inputs),
        seed=int(args.seed),
        allow_all_zero_output=bool(args.allow_all_zero_output),
        backend_overrides=backend_overrides,
        mode=int(args.mode),
        output_path=args.out,
        output_mode=output_mode,
        timeout_s=timeout_s,
        do_clean=bool(args.do_clean),
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cli.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_lp = sub.add_parser("list-presets")
    ap_lp.set_defaults(fn=lambda _a: _cmd_list_presets())

    ap_lt = sub.add_parser("list-tests")
    ap_lt.add_argument("yaml_path")
    ap_lt.add_argument("--test-ids", default=None, help="Semicolon-separated id patterns (supports trailing *)")
    ap_lt.set_defaults(fn=_cmd_list_tests)

    ap_vy = sub.add_parser("validate-yaml")
    ap_vy.add_argument("yaml_path")
    ap_vy.add_argument("--test-ids", default=None, help="Semicolon-separated id patterns (supports trailing *)")
    ap_vy.set_defaults(fn=_cmd_validate_yaml)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("yaml_path")
    ap_run.add_argument("--backends", required=True, help="Semicolon-separated backend preset names: A;B;C")
    ap_run.add_argument("--baseline", default=None)
    ap_run.add_argument("--test-ids", default=None, help="Semicolon-separated id patterns (supports trailing *)")
    ap_run.add_argument("--skips", default=None, help="Semicolon-separated skips: id:backend;id:backend (backend can be *)")
    ov = ap_run.add_mutually_exclusive_group()
    ov.add_argument("--backend-overrides-file", default=None, help="JSON/YAML file for backend_overrides dict")
    ov.add_argument("--backend-overrides-json", default=None, help="Inline JSON string for backend_overrides dict")
    ap_run.add_argument("--repeats", type=int, default=1)
    ap_run.add_argument("--num-sampled-inputs", type=int, default=1)
    ap_run.add_argument("--seed", type=int, default=0)
    ap_run.add_argument("--mode", type=int, default=1, choices=[1, 2])
    ap_run.add_argument("--timeout-s", type=float, default=None)
    ap_run.add_argument("--out", required=True)
    ap_run.add_argument("--append", action="store_true", default=False)
    ap_run.add_argument("--allow-all-zero-output", dest="allow_all_zero_output", action="store_true", default=True)
    ap_run.add_argument("--no-allow-all-zero-output", dest="allow_all_zero_output", action="store_false")
    ap_run.add_argument("--clean-tmp", dest="do_clean", action="store_true", default=True)
    ap_run.add_argument("--no-clean-tmp", dest="do_clean", action="store_false")
    ap_run.set_defaults(fn=_cmd_run)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as e:
        msg = str(e) or repr(e)
        sys.stderr.write(msg + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
