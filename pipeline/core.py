from typing import Any, Dict, List, Optional, Tuple

import torch
from pipeline.utils import clean_sys_tmp
from pipeline.presets import make_backend
from testgen.test_loader import load_config
from testgen.test_plan import extract_test_specs
from testgen.test_input_gen import InputGen
from pipeline.utils import logger, assert_out_not_all_zero, silence_output
from measure.metrics import diff_metrics

def _match_id(pattern: str, actual_id: str) -> bool:
    """Check whether a test id matches a pattern with optional trailing-wildcard prefix support."""
    p = str(pattern)
    a = str(actual_id)
    if p.endswith("*"):
        return a.startswith(p[:-1])
    return a == p


def resolve_tests(yaml_path: str, *, test_ids: Optional[List[str]] = None) -> Tuple[dict, List[dict]]:
    """Load a YAML test configuration and select concrete test specs by id or pattern.

    This function reads the test configuration, expands all test specifications, and optionally
    filters them using exact or prefix-matching test ids. It validates that every requested
    pattern matches at least one test and returns both the full config and the selected tests.

    Args:
      yaml_path: Path to the YAML file defining test specifications.
      test_ids: Optional list of test id patterns to select. A trailing "*" indicates
        prefix matching; otherwise ids must match exactly.

    Returns:
      A tuple of:
        - The parsed YAML configuration dictionary.
        - A list of selected test specification dictionaries.

    Raises:
      KeyError: If one or more requested test id patterns do not match any test in the YAML file.
    """
    cfg = load_config(yaml_path)
    tests = list(extract_test_specs(cfg))
    if test_ids:
        patterns = [str(x) for x in test_ids]

        filtered: List[dict] = []
        for t in tests:
            tid = str(t.get("id", ""))
            if any(_match_id(p, tid) for p in patterns):
                filtered.append(t)

        # Validate each requested pattern matches at least one expanded id
        missing: List[str] = []
        all_ids = [str(t.get("id", "")) for t in tests]
        for p in patterns:
            if not any(_match_id(p, tid) for tid in all_ids):
                missing.append(p)
        if missing:
            raise KeyError(f"test_id not found in {yaml_path}: {sorted(missing)}")

        tests = filtered

    return cfg, tests


def _use_fp16_io(preset_name: str) -> bool:
    """Return True if the given backend preset should use fp16-prepared inputs/exports."""
    return preset_name.endswith("_fp16") and preset_name.split("_", 1)[0] in {
        "torch", "onnx", "executorch"
    }


def _prepare_one(gen: InputGen, cfg: dict, test: dict, *, num_sampled_inputs: int = 1) -> Dict[str, Any]:
    """Prepare a single test case's op name and fp32/fp16 input bundles for execution/export.

    `InputGen.prepare_test_run(..., k=...)` now always returns lists (length k, even when k==1),
    so we store eager/export bundles as lists for both fp32 and fp16.
    """
    k = int(num_sampled_inputs)
    op_name, eager_fp32, export_fp32 = gen.prepare_test_run(cfg, test, k=k)
    
    # eager_fp32/export_fp32 are lists of length k
    eager_fp16 = gen.cast_precision(eager_fp32, dtype=torch.float16)
    export_fp16 = gen.cast_precision(export_fp32, dtype=torch.float16)

    return dict(
        op_name=op_name,
        eager_fp32=eager_fp32,
        export_fp32=export_fp32,
        eager_fp16=eager_fp16,
        export_fp16=export_fp16,
    )

def _ms(ms_str: Optional[str]) -> Optional[float]:
    """Parse a millisecond string into a float, returning None if unavailable/invalid."""
    if ms_str is None:
        return None
    try:
        return float(ms_str)
    except Exception:
        return None
    

def _export_model(
    backend: Any,
    *,
    op: Any,
    export_bundle: Dict[str, Any],
) -> Any:
    """Create a runnable backend model for an op and return the inference inputs/kwargs."""
    if backend.backend_name() == "torch":
        with silence_output(True):
            model = backend.export_op(op)  # no op
        return model
    with silence_output(True):
        model = backend.export_op(
            op=op,
            example_tensor_inputs=export_bundle["tensor_args"],
            const_args=export_bundle["const_args"],
            arg_is_tensor=export_bundle["arg_is_tensor"],
            const_kwargs=export_bundle["const_kwargs"],
            kw_tensor_keys=export_bundle["kw_tensor_keys"],
            out_path=None,
            device=export_bundle["device"],
            cast_input0_to_complex=export_bundle["cast_input0_to_complex"],
        )
    return model

def _infer_once(backend, model, infer_inputs, infer_kwargs):
    """Run a single inference on a backend model using optional keyword arguments."""
    if infer_kwargs is not None:
        return backend.predict(model, infer_inputs, kwargs=infer_kwargs)
    return backend.predict(model, infer_inputs)


def run_backend(
    backend,
    *,
    preset_name: str,
    op: Any,
    eager_inputs: List[Dict[str, Any]],
    export_bundle: List[Dict[str, Any]],
    repeats: int,
    allow_all_zero_output: bool,
    mode: int = 1,
) -> Tuple[str, str, Any, str, Optional[Dict[str, Any]]]:
    """Export the op once and run repeated inferences on the backend while cycling through
    pre-sampled inputs, returning the first output, a median inference time, and optional
    per-repeat outputs/timings (mode=2), with coarse error status buckets on failure."""

    error_max_len = 200
    
    k = len(eager_inputs)
    model = None
    out = None
    ms_str = "x"
    extra = None
    # export once (use sample 0)
    try:
        model = _export_model(
            backend, op=op, export_bundle=export_bundle[0]
        )
    except Exception as e:
        msg = (str(e) or repr(e))[:error_max_len]
        return "EXPORT_FAIL", "x", None, f"{type(e).__name__}: {msg}", None

    # run repeats inferences, cycling through samples
    try:
        times = []
        outs_all = [] if mode == 2 else None
        
        for i in range(int(repeats)):
            j = i % k
            ei = eager_inputs[j]
            bi = export_bundle[j]

            # Match existing backend-specific calling convention:
            if backend.backend_name() == "torch":
                infer_inputs = ei["args"]
                infer_kwargs = ei["kwargs"]
            else:
                infer_inputs = bi["tensor_args"]
                infer_kwargs = None
            # print (backend.backend_name(), infer_inputs)
            cur_out = _infer_once(backend, model, infer_inputs, infer_kwargs)
            # print (backend.backend_name(), cur_out)
            if i == 0:
                out = cur_out
                if not allow_all_zero_output:
                    assert_out_not_all_zero(out, op_name=str(op), preset_name=preset_name)
            if mode == 2:
                outs_all.append(cur_out)
            times.append(float(backend.get_runtime_stats_by_key("infer")))

        times_sorted = sorted(times)
        med_ms = times_sorted[len(times_sorted) // 2] if times_sorted else float("nan")
        ms_str = f"{med_ms:.4f}"
        
        if mode == 2:
            extra = {"times": times, "outs": outs_all}

    except Exception as e:
        msg = (str(e) or repr(e))[:error_max_len]
        # Keep the same coarse status buckets as before
        if out is None:
            return "INFER_FAIL", "x", None, f"{type(e).__name__}: {msg}", None
        return "TIMING_FAIL", "x", out, f"{type(e).__name__}: {msg}", None

    return "OK", ms_str, out, "", extra


def build_test_items(
    *,
    yaml_path: Optional[str],
    test_ids: Optional[List[str]] = None,
    skipped_tests: Optional[List[Dict[str, str]]] = None,  # [{"id", "backend"}]
    adhoc_pairs: Optional[List[Dict[str, str]]] = None,  # [{"left_yaml","left_id","right_yaml","right_id","id"?(opt)}]
):
    """Construct concrete test items (single or pair) from YAML specs and/or ad-hoc pairs,
    applying test-id selection and skip rules, and returning the runnable items along with
    the resolved set of skipped (id, backend) combinations."""
    skipped_set = set()
    if skipped_tests:
        for s in skipped_tests:
            skipped_set.add((str(s["id"]), str(s["backend"])))

    items: List[Dict[str, Any]] = []
    if yaml_path is not None:
        cfg, tests = resolve_tests(yaml_path, test_ids=test_ids)
        for t in tests:
            if t.get("type") == "pair":
                a_test = t["a"]
                b_test = t["b"]
                item_id = str(t.get("id", "")) or f"{a_test.get('id','')}_vs_{b_test.get('id','')}"
                items.append(
                    {
                        "type": "pair",
                        "id": item_id,
                        "sides": [
                            {"yaml": yaml_path, "id": str(a_test.get("id", "")), "test": a_test, "cfg": cfg},
                            {"yaml": yaml_path, "id": str(b_test.get("id", "")), "test": b_test, "cfg": cfg},
                        ],
                    }
                )
            else:
                items.append(
                    {
                        "type": "single",
                        "id": str(t.get("id", "")),
                        "sides": [{"yaml": yaml_path, "id": str(t.get("id", "")), "test": t, "cfg": cfg}],
                    }
                )

    if adhoc_pairs:
        for p in adhoc_pairs:
            left_yaml = p["left_yaml"]
            left_id = p["left_id"]
            right_yaml = p["right_yaml"]
            right_id = p["right_id"]
            item_id = p.get("id") or f"{left_id}_vs_{right_id}"

            lcfg, ltests = resolve_tests(left_yaml, test_ids=[left_id])
            lt = ltests[0]
            if lt.get("type") == "pair":
                raise ValueError(f"ad hoc left_id must be non-pair test (got pair): {left_yaml}::{left_id}")

            rcfg, rtests = resolve_tests(right_yaml, test_ids=[right_id])
            rt = rtests[0]
            if rt.get("type") == "pair":
                raise ValueError(f"ad hoc right_id must be non-pair test (got pair): {right_yaml}::{right_id}")

            items.append(
                {
                    "type": "pair",
                    "id": item_id,
                    "sides": [
                        {"yaml": left_yaml, "id": str(left_id), "test": lt, "cfg": lcfg},
                        {"yaml": right_yaml, "id": str(right_id), "test": rt, "cfg": rcfg},
                    ],
                }
            )
    return items, skipped_set


def _backend_record_skipped(preset_name: str) -> Dict[str, Any]:
    return {
        "backend": preset_name,
        "res": [{"status": "SKIPPED", "err": None, "ms": None}],
        "times_all": None,
        "diff_all": None,
    }

def prepare_item(
    *,
    item: Dict[str, Any],
    seed: int,
    num_sampled_inputs: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Prepare per-side eager/export bundles once and build the out_item shell."""
    sides = item["sides"]
    prepared_sides: List[Dict[str, Any]] = []
    for side_idx, s in enumerate(sides):
        side_seed = int(seed) + (1 if side_idx == 1 else 0)  # right side gets seed+1
        gen = InputGen(seed=int(side_seed))

        prep = _prepare_one(
            gen,
            s["cfg"],
            s["test"],
            num_sampled_inputs=int(num_sampled_inputs),
        )
        prepared_sides.append({"yaml": s["yaml"], "id": s["id"], "test": s["test"], "prep": prep})

    out_item: Dict[str, Any] = {
        "id": str(item.get("id", "")),
        "type": item["type"],
        "cases": [{"yaml": s["yaml"], "id": s["id"], "op": str(s["prep"]["op_name"])} for s in prepared_sides],
        "baseline": None,
        "backends": [],
    }
    return prepared_sides, out_item

def run_baseline_if_needed(
    *,
    out_item: Dict[str, Any],
    prepared_sides: List[Dict[str, Any]],
    baseline_backend: Optional[str],
    overrides_by_backend: Dict[str, Dict[str, Any]],
    skipped_set: set,
    repeats: int,
    allow_all_zero_output: bool,
    mode: int,
) -> Tuple[Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]]:
    """
    Single-item only baseline run.
    Returns: (baseline_info, baseline_out, baseline_extra)
    """
    baseline_info = None
    baseline_out = None
    baseline_extra = None
    
    if baseline_backend is None or len(prepared_sides) != 1:
        return None, None, None
    
    base_name = baseline_backend
    
    if (out_item["id"], base_name) in skipped_set or (out_item["id"], "*") in skipped_set:
        baseline_info = {"backend": base_name, "status": "SKIPPED", "err": None, "ms": None}
        out_item["baseline"] = baseline_info
        return baseline_info, None, None
    
    base_backend = make_backend(base_name, **overrides_by_backend.get(base_name, {}))

    prep0 = prepared_sides[0]["prep"]
    op0 = prep0["op_name"]

    if _use_fp16_io(base_name):
        eager0 = prep0["eager_fp16"]   
        bundle0 = prep0["export_fp16"] 
    else:
        eager0 = prep0["eager_fp32"]
        bundle0 = prep0["export_fp32"]

    b_status, b_ms_str, b_out, b_err, b_extra = run_backend(
        base_backend,
        preset_name=base_name,
        op=op0,
        eager_inputs=eager0,
        export_bundle=bundle0,
        repeats=int(repeats),
        allow_all_zero_output=allow_all_zero_output,
        mode=mode,
    )

    baseline_info = {
        "backend": base_name,
        "status": b_status,
        "err": (b_err or None),
        "ms": _ms(b_ms_str),
    }
    
    if mode == 2 and b_status == "OK" and b_extra is not None:
        baseline_info["times_all"] = [b_extra.get("times")]
    else:
        baseline_info["times_all"] = None
    
    baseline_out = b_out if b_status == "OK" else None
    baseline_extra = b_extra if b_status == "OK" else None
    out_item["baseline"] = baseline_info
    logger.info("FINISHED id=%s backend=%s status=OK", out_item["id"], base_name)
    return baseline_info, baseline_out, baseline_extra

def run_item_on_backend(
    *,
    item: Dict[str, Any],
    out_item_id: str,
    prepared_sides: List[Dict[str, Any]],
    preset_name: str,
    baseline_backend: Optional[str],
    baseline_info: Optional[Dict[str, Any]],
    baseline_out: Any,
    baseline_extra: Optional[Dict[str, Any]],
    overrides_by_backend: Dict[str, Dict[str, Any]],
    skipped_set: set,
    repeats: int,
    allow_all_zero_output: bool,
    mode: int,
) -> Dict[str, Any]:
    """Run one item on one backend and produce the exact backend record dict."""
    if (out_item_id, preset_name) in skipped_set or (out_item_id, "*") in skipped_set:
        logger.info("FINISHED id=%s backend=%s status=SKIPPED", out_item_id, preset_name)
        return _backend_record_skipped(preset_name)

    backend = make_backend(preset_name, **overrides_by_backend.get(preset_name, {}))

    side_recs: List[Dict[str, Any]] = []
    side_outs: List[Any] = []
    side_extras: List[Optional[Dict[str, Any]]] = []

    for s in prepared_sides:
        prep = s["prep"]
        op_name = prep["op_name"]

        if _use_fp16_io(preset_name):
            eager_inputs = prep["eager_fp16"]  
            export_bundle = prep["export_fp16"]
        else:
            eager_inputs = prep["eager_fp32"]
            export_bundle = prep["export_fp32"]

        status, ms_str, run_out, err, extra = run_backend(
            backend,
            preset_name=preset_name,
            op=op_name,
            eager_inputs=eager_inputs,
            export_bundle=export_bundle,
            repeats=int(repeats),
            allow_all_zero_output=allow_all_zero_output,
            mode=mode,
        )

        rec: Dict[str, Any] = {"status": status, "err": (err or None), "ms": _ms(ms_str)}
        side_recs.append(rec)
        side_outs.append(run_out if status == "OK" else None)
        side_extras.append(extra if status == "OK" else None)

    b: Dict[str, Any] = {
        "backend": preset_name,
        "res": side_recs,
        "times_all": None, 
        "diff_all": None, 
    }

    if mode == 2:
        b["times_all"] = [(ex["times"] if ex is not None else None) for ex in side_extras]

    if item["type"] == "pair":
        if side_recs[0]["status"] == "OK" and side_recs[1]["status"] == "OK":
            if mode == 1:
                mse, mx, mean, code, msg = diff_metrics(side_outs[0], side_outs[1])
                if code == "NONE":
                    b["diff_all"] = [{"ok": True, "mse": float(mse), "max_abs": float(mx), "mean_abs": float(mean)}]
                else:
                    b["diff_all"] = [{"ok": False, "code": code, "msg": msg}]
            else:
                # mode 2: per-repeat diff: outsA[i] vs outsB[i]
                outsA = side_extras[0]["outs"] if side_extras[0] is not None else None
                outsB = side_extras[1]["outs"] if side_extras[1] is not None else None
                if outsA is None or outsB is None:
                    b["diff_all"] = None
                else:
                    n = min(len(outsA), len(outsB), int(repeats))
                    diff_all = []
                    for i in range(n):
                        mse, mx, mean, code, msg = diff_metrics(outsA[i], outsB[i])
                        if code == "NONE":
                            diff_all.append({"ok": True, "mse": float(mse), "max_abs": float(mx), "mean_abs": float(mean)})
                        else:
                            diff_all.append({"ok": False, "code": code, "msg": msg})
                    b["diff_all"] = diff_all
        else:
            b["diff_all"] = None
    else:
        # single: compare to baseline if provided & OK
        if baseline_backend is not None and baseline_info is not None:
            if baseline_info["status"] == "OK" and side_recs[0]["status"] == "OK":
                if mode == 1:
                    mse, mx, mean, code, msg = diff_metrics(baseline_out, side_outs[0])
                    if code == "NONE":
                        b["diff_all"] = [{"ok": True, "mse": float(mse), "max_abs": float(mx), "mean_abs": float(mean)}]
                    else:
                        b["diff_all"] = [{"ok": False, "code": code, "msg": msg}]
                else:
                    # mode 2: baseline outs[i] vs target outs[i]
                    base_outs = baseline_extra["outs"] if baseline_extra is not None else None
                    targ_outs = side_extras[0]["outs"] if side_extras[0] is not None else None
                    if base_outs is None or targ_outs is None:
                        b["diff_all"] = None
                    else:
                        n = min(len(base_outs), len(targ_outs), int(repeats))
                        diff_all = []
                        for i in range(n):
                            mse, mx, mean, code, msg = diff_metrics(base_outs[i], targ_outs[i])
                            if code == "NONE":
                                diff_all.append({"ok": True, "mse": float(mse), "max_abs": float(mx), "mean_abs": float(mean)})
                            else:
                                diff_all.append({"ok": False, "code": code, "msg": msg})
                        b["diff_all"] = diff_all
            else:
                b["diff_all"] = None
        else:
            b["diff_all"] = None
    # clean_sys_tmp()
    return b

def run_one_item(
    *,
    item: Dict[str, Any],
    seed: int,
    num_sampled_inputs: int,
    baseline_backend: Optional[str],
    target_backends: List[str],
    repeats: int,
    allow_all_zero_output: bool,
    overrides_by_backend: Dict[str, Dict[str, Any]],
    skipped_set: set,
    mode: int = 1,
) -> Dict[str, Any]:
    """Execute one resolved test item (single or pair) across backends and return its result record.

    For single items, an optional baseline backend is run once and used for diffs; for pair items,
    diffs are computed between left and right sides under the same backend. In mode=2, per-repeat
    timings/outputs and per-repeat diffs are recorded.
    """
    prepared_sides, out_item = prepare_item(
        item=item, seed=seed, num_sampled_inputs=int(num_sampled_inputs)
    )

    # Baseline (single only; uses the prepared inputs for side 0)
    baseline_info, baseline_out, baseline_extra = run_baseline_if_needed(
        out_item=out_item,
        prepared_sides=prepared_sides,
        baseline_backend=baseline_backend,
        overrides_by_backend=overrides_by_backend,
        skipped_set=skipped_set,
        repeats=int(repeats),
        allow_all_zero_output=allow_all_zero_output,
        mode=mode,
    )

    # Backends
    for preset_name in target_backends:
        b = run_item_on_backend(
            item=item,
            out_item_id=out_item["id"],
            prepared_sides=prepared_sides,
            preset_name=preset_name,
            baseline_backend=baseline_backend,
            baseline_info=baseline_info,
            baseline_out=baseline_out,
            baseline_extra=baseline_extra,
            overrides_by_backend=overrides_by_backend,
            skipped_set=skipped_set,
            repeats=int(repeats),
            allow_all_zero_output=allow_all_zero_output,
            mode=mode,
        )
        out_item["backends"].append(b)
        logger.info("FINISHED id=%s backend=%s status=OK", out_item["id"], preset_name)
    return out_item