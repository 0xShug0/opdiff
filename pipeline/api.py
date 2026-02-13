from __future__ import annotations

from typing import Any, Dict, List, Optional
import multiprocessing as mp
from measure.jsonl_writter import JSONLWriter
from pipeline.core import _backend_record_skipped, build_test_items, prepare_item, run_baseline_if_needed, run_item_on_backend, run_one_item
from pipeline.utils import logger, clean_sys_tmp


def _run_one_backend_worker_timed(
    q,
    *,
    item,
    out_item_id: str,
    prepared_sides,
    backend_name: str,
    baseline_backend,
    baseline_info,
    baseline_out,
    baseline_extra,
    repeats: int,
    allow_all_zero_output: bool,
    overrides_by_backend,
    skipped_set,
    mode: int,
):
    """
    Worker runs exactly ONE (item, backend) and returns a single backend record `b`.
    """
    try:
        b = run_item_on_backend(
            item=item,
            out_item_id=out_item_id,
            prepared_sides=prepared_sides,
            preset_name=backend_name,
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
        q.put(("OK", b))
    except BaseException as e:
        q.put(("ERROR", {"type": type(e).__name__, "msg": str(e) or repr(e)}))


def run_timed(
    *,
    yaml_path: str,
    target_backends: List[str],
    timeout_s: float,
    test_ids: Optional[List[str]] = None,
    skipped_tests: Optional[List[Dict[str, str]]] = None,
    baseline_backend: Optional[str] = None,
    adhoc_pairs: Optional[List[Dict[str, str]]] = None,
    repeats: int = 1,
    num_sampled_inputs: int = 1,
    seed: int = 0,
    allow_all_zero_output: bool = True,
    backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: int = 1,
    output_path: str = "",
    output_mode: str = "w",
) -> None:
    """Execute a backend diff run with per-(item, backend) hard timeouts.

    Each backend execution is isolated in a subprocess and terminated if it
    exceeds `timeout_s`. Failures and timeouts are recorded as TIMED_ERROR
    without stopping the overall run. Results are streamed to JSONL.
    """
    
    overrides_by_backend = backend_overrides or {}
    params = {
        "seed": seed,
        "repeats": repeats,
        "num_sampled_inputs": int(num_sampled_inputs),
        "allow_all_zero_output": allow_all_zero_output,
        "backend_overrides": backend_overrides,
        "mode": mode,
        "timeout_s": timeout_s,
    }

    writer = JSONLWriter(output_path)
    writer.start_run(run_info={"yaml": yaml_path, "params": params}, mode=output_mode)

    items, skipped_set = build_test_items(
        yaml_path=yaml_path,
        test_ids=test_ids,
        skipped_tests=skipped_tests,
        adhoc_pairs=adhoc_pairs,
    )

    ctx = mp.get_context("spawn")

    for item in items:
        prepared_sides, out_item = prepare_item(
            item=item,
            seed=seed,
            num_sampled_inputs=int(num_sampled_inputs),
        )
        
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

        # Timed per-backend runs
        for backend_name in target_backends:
            if (out_item["id"], backend_name) in skipped_set or (out_item["id"], "*") in skipped_set:
                out_item["backends"].append(_backend_record_skipped(backend_name))
                logger.info("FINISHED id=%s backend=%s status=SKIPPED", out_item["id"], backend_name)
                continue

            q = ctx.Queue()
            p = ctx.Process(
                target=_run_one_backend_worker_timed,
                kwargs=dict(
                    q=q,
                    item=item,
                    out_item_id=out_item["id"],
                    prepared_sides=prepared_sides,
                    backend_name=backend_name,
                    baseline_backend=baseline_backend,
                    baseline_info=baseline_info,
                    baseline_out=baseline_out,
                    baseline_extra=baseline_extra,
                    repeats=int(repeats),
                    allow_all_zero_output=allow_all_zero_output,
                    overrides_by_backend=overrides_by_backend,
                    skipped_set=skipped_set,
                    mode=mode,
                ),
                daemon=True,
            )

            p.start()
            p.join(timeout=timeout_s)

            if p.is_alive():
                p.terminate()
                p.join()
                out_item["backends"].append(
                    {
                        "backend": backend_name,
                        "res": [{"status": "TIMED_ERROR", "err": f"TIMEOUT: timeout_s={timeout_s}", "ms": None}],
                        "times_all": None,
                        "diff_all": None,
                    }
                )
                logger.info("FINISHED id=%s backend=%s status=TIMEOUT", out_item["id"], backend_name)
                continue

            try:
                tag, payload = q.get_nowait()
            except Exception:
                out_item["backends"].append(
                    {
                        "backend": backend_name,
                        "res": [{"status": "TIMED_ERROR", "err": f"CRASH: exitcode={p.exitcode}", "ms": None}],
                        "times_all": None,
                        "diff_all": None,
                    }
                )
                logger.info("FINISHED id=%s backend=%s status=CRASH", out_item["id"], backend_name)
                continue

            if tag == "OK":
                out_item["backends"].append(payload)
                logger.info("FINISHED id=%s backend=%s status=OK", out_item["id"], backend_name)
            else:
                out_item["backends"].append(
                    {
                        "backend": backend_name,
                        "res": [{"status": "TIMED_ERROR", "err": payload or None, "ms": None}],
                        "times_all": None,
                        "diff_all": None,
                    }
                )
                logger.info("FINISHED id=%s backend=%s status=TIMED_ERROR", out_item["id"], backend_name)
                

        writer.write(out_item)

def run_inprocess(
    *,
    yaml_path: str,
    target_backends: List[str],
    test_ids: Optional[List[str]] = None,
    skipped_tests: Optional[List[Dict[str, str]]] = None,
    baseline_backend: Optional[str] = None,
    adhoc_pairs: Optional[List[Dict[str, str]]] = None, 
    repeats: int = 1,
    num_sampled_inputs: int = 1,
    seed: int = 0,
    allow_all_zero_output: bool = True,
    backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: int = 1,
    output_path: str = "",
    output_mode: str = "w", 
) -> None:
    """Execute a backend diff run entirely in-process.

    For each selected item and backend, export once, run repeated inferences,
    compute median latency and output diffs, and stream results to JSONL.
    Exceptions propagate normally and may terminate the run.
    """
    
    overrides_by_backend = backend_overrides or {}
    params = {
        "seed": seed,
        "repeats": repeats,
        "num_sampled_inputs": int(num_sampled_inputs),
        "allow_all_zero_output": allow_all_zero_output,
        "backend_overrides": backend_overrides,
        "mode": mode,
    }
    
    writer = JSONLWriter(output_path)
    writer.start_run(run_info={"yaml": yaml_path, "params": params}, mode=output_mode)
                
    items, skipped_set = build_test_items(
        yaml_path=yaml_path,
        test_ids=test_ids,
        skipped_tests=skipped_tests,
        adhoc_pairs=adhoc_pairs,
    )
    
    for item in items:
        out_item = run_one_item(
            item=item,
            seed=seed,
            num_sampled_inputs=int(num_sampled_inputs),
            baseline_backend=baseline_backend,
            target_backends=target_backends,
            repeats=int(repeats),
            allow_all_zero_output=allow_all_zero_output,
            overrides_by_backend=overrides_by_backend,
            skipped_set=skipped_set,
            mode=mode,
        )        
        writer.write(out_item)



def run(
    *,
    yaml_path: str,
    target_backends: List[str],
    test_ids: Optional[List[str]] = None,
    skipped_tests: Optional[List[Dict[str, str]]] = None,
    baseline_backend: Optional[str] = None,
    adhoc_pairs: Optional[List[Dict[str, str]]] = None,
    repeats: int = 1,
    num_sampled_inputs: int = 1,
    seed: int = 0,
    allow_all_zero_output: bool = True,
    backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: int = 1,
    output_path: str = "",
    output_mode: str = "w",
    timeout_s: Optional[float] = None,
    do_clean: bool = True,
) -> None:
    """Execute a deterministic backend comparison run and write per-test results to JSONL.

    This is the unified entrypoint for backend diff runs. Depending on whether
    `timeout_s` is provided, execution is performed either:

      - in-process (standard mode), or
      - in isolated subprocesses with a hard per-(item, backend) timeout.

    The run operates over a set of test items derived from a YAML test suite
    (`yaml_path`) with optional id filtering. Each item may represent either:
      - a single test case, or
      - a pair of test cases compared under the same backend.

    For each selected item, the runner:
      - deterministically samples `num_sampled_inputs` independent input sets
        (seeded by `seed`, and `seed+1` for the right side of pair tests),
      - exports each backend once per side using the first sampled input,
      - executes `repeats` inferences while cycling through sampled inputs,
      - records a robust latency statistic (median inference time), and
      - computes output diffs:
            * pair items: left vs right outputs under the same backend
            * single items: each target backend vs an optional baseline backend

    Results are streamed to a JSONL file via `JSONLWriter`, including a run
    header record with configuration metadata followed by one record per item.

    Args:
        yaml_path: Path to a YAML file describing tests to run.
        target_backends: Backend preset names to execute for each item
            (e.g. "torch_cpu_fp32", "coreml_cpu_fp32", "onnx_cpu_fp32").
        test_ids: Optional list of test id patterns to select from the YAML suite.
            A trailing "*" indicates prefix matching; otherwise ids must match exactly.
        skipped_tests: Optional list of {"id", "backend"} entries to skip running
            specific backends for specific test ids.
        baseline_backend: Optional backend preset used as the baseline for
            single-item tests. Ignored for pair tests.
        adhoc_pairs: Optional explicit left-vs-right comparisons.
        repeats: Number of inference repetitions per backend execution.
        num_sampled_inputs: Number of independent sampled input sets to prepare
            per side. Repeats cycle through these inputs with modulo indexing.
        seed: Base RNG seed for deterministic input generation.
        allow_all_zero_output: If False, raises an error when the first observed
            backend output is entirely zero-valued.
        backend_overrides: Optional per-backend configuration dict passed to
            `make_backend`.
        mode:
            - 1: compute a single diff per comparison (based on first output)
            - 2: compute per-repeat diffs and record per-repeat timings/outputs
        output_path: Path to the JSONL output file.
        output_mode:
            - "w": start a fresh run (overwrite)
            - "a": append a new run header and subsequent item records
        timeout_s: If provided, enforce a hard timeout (seconds) per
            (item, backend) using subprocess isolation. If None, run in-process.
        clean_sys_tmp: If True, clean system temp and cache after the run completes.

    Notes:
        - Latency is reported as the median of backend-reported per-infer timings.
        - Export is performed once per backend per side using the first sampled input.
        - In `mode=2`, timings and outputs are collected per repeat and diffs are
          computed using aligned repeat indices.
        - When `timeout_s` is set, backend crashes and timeouts are reported as
          TIMED_ERROR statuses without affecting other backends.
    """
    try:
        if timeout_s is not None:
            run_timed(
                yaml_path=yaml_path,
                target_backends=target_backends,
                timeout_s=float(timeout_s),
                test_ids=test_ids,
                skipped_tests=skipped_tests,
                baseline_backend=baseline_backend,
                adhoc_pairs=adhoc_pairs,
                repeats=int(repeats),
                num_sampled_inputs=int(num_sampled_inputs),
                seed=int(seed),
                allow_all_zero_output=allow_all_zero_output,
                backend_overrides=backend_overrides,
                mode=mode,
                output_path=output_path,
                output_mode=output_mode,
            )
            return

        run_inprocess(
            yaml_path=yaml_path,
            target_backends=target_backends,
            test_ids=test_ids,
            skipped_tests=skipped_tests,
            baseline_backend=baseline_backend,
            adhoc_pairs=adhoc_pairs,
            repeats=int(repeats),
            num_sampled_inputs=int(num_sampled_inputs),
            seed=int(seed),
            allow_all_zero_output=allow_all_zero_output,
            backend_overrides=backend_overrides,
            mode=mode,
            output_path=output_path,
            output_mode=output_mode,
        )
    finally:
        if do_clean:
            clean_sys_tmp()
            
__all__ = ["run"]