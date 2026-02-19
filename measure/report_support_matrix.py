#!/usr/bin/env python3
# Produce backend support/robustness matrix (CSV + heatmap + short markdown report)
#
# Usage:
#   python support_matrix.py path/to/results.jsonl
#   python support_matrix.py path/to/results.jsonl --out report_support_matrix
#
# Outputs (under --out):
#   - matrix_wide.csv
#   - support_heatmap.png
#   - report.md
#
# Notes:
# - Parses JSONL emitted by jsonl_writter.JSONLWriter:
#   * meta record: {"type":"meta","meta":{...}}
#   * item record: {"type":"item","id":..., "test_type":..., "cases":..., "baseline":..., "backends":[...]}

import os
import re
import sys
import json
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEVERITY = {
    # higher = worse
    "OK": 0,
    "SKIPPED": 1,

    "EXPORT_FAIL": 2,
    "INFER_FAIL": 3,
    "TIMING_FAIL": 4,

    "TIMED_ERROR": 5,

    "ERROR": 6,
}

# Literal buckets (keep "<n>" and "[op]" exactly as text)
_BUCKET_RULES = [
    # coreml
    (lambda s: ("ValueError: Torch var" in s and "not found in context" in s),
     "ValueError Torch var <n> not found in context"),

    # NotImplementedError family (order matters)
    (lambda s: (s.startswith("NotImplementedError:") and
                "The operator" in s and
                "is not currently implemented for the MPS device" in s),
     "NotImplementedError: The operator [op] is not currently implemented for the MPS device"),
    (lambda s: (s.startswith("NotImplementedError:") and
                "Could not run" in s and
                "with arguments from the 'CPU' backend" in s),
     "NotImplementedError: Could not run [op] with arguments from the 'CPU' backend"),
    (lambda s: s.startswith("NotImplementedError:"),
     "NotImplementedError"),

    (lambda s: "AttributeError: type object 'Builder' has no attribute" in s,
     "AttributeError: type object 'Builder' has no attribute [op]"),

    # onnx
    (lambda s: "ConversionError: Failed to convert the exported program to an ONNX model" in s,
     "ConversionError: Failed to convert the exported program to an ONNX model"),
    (lambda s: "TorchExportError: Failed to export the model with torch.export" in s,
     "TorchExportError: Failed to export the model with torch.export"),
    (lambda s: s.startswith("PassError:"),
     "PassError"),
    (lambda s: s.startswith("Fail: [ONNXRuntimeError]"),
     "Fail: [ONNXRuntimeError]"),
    (lambda s: s.startswith("InvalidGraph: [ONNXRuntimeError]"),
     "InvalidGraph: [ONNXRuntimeError]"),
    (lambda s: "ConversionError: Failed to decompose the FX graph for ONNX compatibility" in s,
     "ConversionError: Failed to decompose the FX graph for ONNX compatibility"),

    # executorch (and similar)
    (lambda s: s.startswith("SpecViolationError"),
     "SpecViolationError"),
    (lambda s: s.startswith("RuntimeError: loading method forward failed"),
     "RuntimeError: loading method forward failed"),
    (lambda s: (s.startswith("RuntimeError: Expected one of cpu, cuda") and "device type at start" in s),
     "RuntimeError: Expected one of cpu, cuda, ... device type at start"),
    (lambda s: s.startswith("RuntimeError: method->execute() failed with error"),
     "RuntimeError: method->execute() failed with error"),

    (lambda s: (s.startswith("Unsupported: torch.") and "op returned non-Tensor" in s),
     "Unsupported: torch.* op returned non-Tensor"),
    (lambda s: s.startswith("Unsupported: Error when attempting to resolve op packet"),
     "Unsupported: Error when attempting to resolve op packet"),
    (lambda s: s.startswith("Unsupported: Data dependent operator"),
     "Unsupported: Data dependent operator"),
    (lambda s: s.startswith("Unsupported: Dynamic shape operator (no meta kernel)"),
     "Unsupported: Dynamic shape operator (no meta kernel)"),
    (lambda s: s.startswith("Unsupported: NotImplementedError/UnsupportedFakeTensorException when running FX node"),
     "Unsupported: NotImplementedError/UnsupportedFakeTensorException when running FX node"),

    (lambda s: s.startswith("TorchRuntimeError: Dynamo failed to run FX node with fake tensors"),
     "TorchRuntimeError: Dynamo failed to run FX node with fake tensors"),
    (lambda s: s.startswith("UserError: Could not guard on data-dependent expression"),
     "UserError: Could not guard on data-dependent expression"),
    (lambda s: s.startswith("GuardOnDataDependentSymNode: Could not guard on data-dependent expression"),
     "GuardOnDataDependentSymNode: Could not guard on data-dependent expression"),

    (lambda s: s.startswith("RuntimeError: Failed to compile"),
     "RuntimeError: Failed to compile"),
    (lambda s: s.startswith("SyntaxError: invalid syntax"),
     "SyntaxError: invalid syntax"),
]


def normalize_err(err_text):
    """
    Map raw err string into one of the user-defined *literal* buckets.
    Returns the bucket string exactly as listed (keeps <n> and [op] literally).
    Unknown messages return "OTHER: <first line truncated>".
    """
    s = (err_text or "").strip()
    if not s:
        return ""

    # first line is enough for bucketing
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()

    # remove raw ESC char (common from colored logs)
    s = s.replace("\x1b", "")

    for pred, bucket in _BUCKET_RULES:
        if pred(s):
            return bucket

    t = s.strip()
    if len(t) > 140:
        t = t[:140] + "…"
    return "OTHER: " + t


def severity_of(status):
    if status is None:
        return 999
    return SEVERITY.get(str(status), 900)


def combine_status(status_list):
    # Choose the "worst" status among sides (single has 1 side, pair has 2 sides)
    if not status_list:
        return "ERROR"
    worst = None
    worst_s = -1
    for st in status_list:
        s = severity_of(st)
        if s > worst_s:
            worst_s = s
            worst = st
    return str(worst)


def is_run_header(obj):
    # jsonl_writter emits:
    #   meta: {"type":"meta","meta":{...}}
    #   item: {"type":"item", ...}
    if not isinstance(obj, dict):
        return True
    t = obj.get("type")
    if t == "meta":
        return True
    if t == "item":
        return False
    # fallback: treat unknown shapes as header/noise
    return "backends" not in obj


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON on line {ln}: {e}")

def _err_to_text(err):
    if err is None:
        return ""
    if isinstance(err, str):
        return err
    try:
        return json.dumps(err, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(err)


def collect_exception_groups(item_ids, backend_cols, per_item, raw_items_by_id):
    """
    Returns:
      per_backend[backend] = Counter(signature -> count)
      global_counter = Counter(signature -> count)
    Notes:
      - Uses *err*, not status.
      - Counts only when err is non-empty.
      - For pair tests, counts both sides if they have err.
      - ALSO scans baseline record (treated as a normal backend).
    """
    per_backend = {b: Counter() for b in backend_cols}
    global_counter = Counter()

    for item_id in item_ids:
        obj = raw_items_by_id.get(item_id)
        if not obj:
            continue
        
        base = obj.get("baseline")
        if isinstance(base, dict):
            backend = str(base.get("backend", "") or "")
            if backend in per_backend:
                err_txt = _err_to_text(base.get("err"))
                if err_txt:
                    sig = normalize_err(err_txt)
                    if sig:
                        per_backend[backend][sig] += 1
                        global_counter[sig] += 1
                        
                res = base.get("res")
                if isinstance(res, list):
                    for r in res:
                        if not isinstance(r, dict):
                            continue
                        err_txt = _err_to_text(r.get("err"))
                        if not err_txt:
                            continue
                        sig = normalize_err(err_txt)
                        if not sig:
                            continue
                        per_backend[backend][sig] += 1
                        global_counter[sig] += 1

        # existing: scan normal backend records
        backends = obj.get("backends") or []
        for b in backends:
            if not isinstance(b, dict):
                continue
            backend = str(b.get("backend", "") or "")
            if backend not in per_backend:
                continue

            res = b.get("res") or []
            if not isinstance(res, list):
                continue

            for r in res:
                if not isinstance(r, dict):
                    continue
                err_txt = _err_to_text(r.get("err"))
                if not err_txt:
                    continue
                sig = normalize_err(err_txt)
                if not sig:
                    continue
                per_backend[backend][sig] += 1
                global_counter[sig] += 1

    return per_backend, global_counter



def build_matrix(objs):
    # rows: item id
    # cols: backend name
    # cell: combined status across sides
    run_meta = None
    all_backend_names = []
    per_item = {}   # item_id -> {backend_name: status}
    item_meta = {}  # item_id -> {"type": single/pair, "op": op_name}
    raw_items_by_id = {}

    for obj in objs:
        if is_run_header(obj):
            if run_meta is None and isinstance(obj, dict) and obj.get("type") == "meta":
                run_meta = obj.get("meta")
            continue

        item_id = str(obj.get("id", "") or "")
        if not item_id:
            continue

        if item_id in raw_items_by_id:
            # merge backend lists only
            raw_items_by_id[item_id]["backends"].extend(obj.get("backends", []))
        else:
            raw_items_by_id[item_id] = obj

        # best-effort op name from cases[0].op
        op_name = ""
        cases = obj.get("cases") or []
        if isinstance(cases, list) and cases and isinstance(cases[0], dict):
            op_name = str(cases[0].get("op", "") or "")

        item_meta[item_id] = {
            "type": str(obj.get("test_type", "") or ""),  # jsonl_writter uses test_type
            "op": op_name,
        }

        row = {}

        # --- NEW: treat baseline as a normal backend column ---
        base = obj.get("baseline")
        if isinstance(base, dict):
            base_name = str(base.get("backend", "") or "")
            if base_name:
                if "status" in base:
                    row[base_name] = str(base.get("status") or "")
                else:
                    res = base.get("res") or []
                    if not isinstance(res, list):
                        res = []
                    statuses = []
                    for r in res:
                        if isinstance(r, dict):
                            statuses.append(r.get("status"))
                    row[base_name] = combine_status(statuses)
                all_backend_names.append(base_name)

        # existing: normal backends
        backends = obj.get("backends") or []
        if not isinstance(backends, list):
            backends = []

        for b in backends:
            if not isinstance(b, dict):
                continue
            backend_name = str(b.get("backend", "") or "")
            if not backend_name:
                continue

            res = b.get("res") or []
            if not isinstance(res, list):
                res = []

            statuses = []
            for r in res:
                if isinstance(r, dict):
                    statuses.append(r.get("status"))

            row[backend_name] = combine_status(statuses)
            all_backend_names.append(backend_name)

        if item_id not in per_item:
            per_item[item_id] = row
        else:
            per_item[item_id].update(row)

    backend_cols = sorted(set(all_backend_names))
    item_ids = sorted(per_item.keys())
    return run_meta, item_ids, backend_cols, per_item, item_meta, raw_items_by_id



def write_matrix_wide_csv(out_path, item_ids, backend_cols, per_item, item_meta):
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "op", "type"] + backend_cols)
        for item_id in item_ids:
            meta = item_meta.get(item_id, {})
            row = per_item.get(item_id, {})
            w.writerow(
                [item_id, meta.get("op", ""), meta.get("type", "")]
                + [row.get(b, "") for b in backend_cols]
            )

def write_report_md(out_path, run_meta, item_ids, backend_cols, per_item, item_meta, raw_items_by_id):
    total_items = len(item_ids)

    per_backend_counts = {b: Counter() for b in backend_cols}
    for item_id in item_ids:
        row = per_item.get(item_id, {})
        for b in backend_cols:
            st = row.get(b, "")
            if st:
                per_backend_counts[b][st] += 1

    def ok_rate(counter):
        denom = sum(counter.values())
        if denom == 0:
            return 0.0
        return 100.0 * counter.get("OK", 0) / denom

    lines = []
    lines.append("# Backend Support / Robustness Report\n")
    lines.append(f"- Items (tests): **{total_items}**")
    lines.append(f"- Backends: **{len(backend_cols)}**\n")

    if isinstance(run_meta, dict):
        lines.append("## Run metadata\n")
        try:
            txt = json.dumps(run_meta, indent=2, ensure_ascii=False)
            if len(txt) > 2000:
                txt = txt[:2000] + "\n... (truncated)\n"
            lines.append("```json")
            lines.append(txt)
            lines.append("```\n")
        except Exception:
            lines.append("_Failed to render run metadata as JSON._\n")

    lines.append("## Per-backend summary\n")
    backend_sorted = sorted(backend_cols, key=lambda b: (-ok_rate(per_backend_counts[b]), b))

    lines.append("| Backend | OK% | OK | SKIPPED | EXPORT_FAIL | INFER_FAIL | TIMING_FAIL | TIMED_ERROR | ERROR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in backend_sorted:
        c = per_backend_counts[b]
        lines.append(
            f"| {b} | {ok_rate(c):.1f} | "
            f"{c.get('OK',0)} | {c.get('SKIPPED',0)} | {c.get('EXPORT_FAIL',0)} | "
            f"{c.get('INFER_FAIL',0)} | {c.get('TIMING_FAIL',0)} | {c.get('TIMED_ERROR',0)} | {c.get('ERROR',0)} |"
        )
    lines.append("")

    lines.append("## Most problematic tests (by non-OK count)\n")
    scores = []
    for item_id in item_ids:
        row = per_item.get(item_id, {})
        bad = 0
        ok = 0
        for b in backend_cols:
            st = row.get(b, "")
            if not st:
                continue
            if st == "OK":
                ok += 1
            else:
                bad += 1
        scores.append((bad, ok, item_id))
    scores.sort(reverse=True)

    lines.append("| Test ID | non-OK | OK | op (best-effort) |")
    lines.append("|---|---:|---:|---|")
    for bad, ok, item_id in scores[:20]:
        op = item_meta.get(item_id, {}).get("op", "")
        lines.append(f"| {item_id} | {bad} | {ok} | {op} |")
    lines.append("")

    
    lines.append("## Exception message groups (from `err`, normalized)\n")

    per_backend_errs, _ = collect_exception_groups(
        item_ids, backend_cols, per_item, raw_items_by_id
    )

    # Per-backend top signatures
    lines.append("### Top exception groups per backend\n")
    for b in backend_cols:
        lines.append(f"#### {b}\n")
        lines.append("| Count | Signature |")
        lines.append("|---:|---|")
        for sig, c in per_backend_errs[b].most_common(15):
            lines.append(f"| {c} | {sig} |")
        if sum(per_backend_errs[b].values()) == 0:
            lines.append("| 0 | (no exceptions captured) |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def make_outcome_stacked_bar_png(out_path, item_ids, backend_cols, per_item):
    # One stacked bar per backend: counts (or %). We’ll plot % for comparability.
    status_order = ["OK", "SKIPPED", "EXPORT_FAIL", "INFER_FAIL", "TIMING_FAIL", "TIMED_ERROR", "ERROR"]

    # counts[backend][status] = n
    counts = {b: Counter() for b in backend_cols}
    denom = {b: 0 for b in backend_cols}

    for item_id in item_ids:
        row = per_item.get(item_id, {})
        for b in backend_cols:
            st = row.get(b, "")
            if not st:
                continue
            if st not in status_order:
                st = "ERROR"
            counts[b][st] += 1
            denom[b] += 1

    # build stacked series (as percentages)
    x = list(range(len(backend_cols)))
    bottoms = [0.0] * len(backend_cols)

    fig_w = max(10, 0.55 * len(backend_cols))
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for st in status_order:
        vals = []
        for i, b in enumerate(backend_cols):
            d = denom[b] or 1
            vals.append(100.0 * counts[b].get(st, 0) / d)

        ax.bar(x, vals, bottom=bottoms, label=st)
        bottoms = [bottoms[i] + vals[i] for i in range(len(bottoms))]

    ax.set_xticks(x)
    ax.set_xticklabels(backend_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Percent of tests (%)")
    ax.set_title("Backend Outcome Composition (stacked %)")
    ax.legend(fontsize=8, ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_backend_overlap_png(out_path, item_ids, backend_cols, per_item):
    # backend×backend matrix: % of tests where both backends are OK
    n = len(backend_cols)
    both_ok = [[0 for _ in range(n)] for _ in range(n)]
    both_seen = [[0 for _ in range(n)] for _ in range(n)]  # tests where both have a status

    for item_id in item_ids:
        row = per_item.get(item_id, {})
        statuses = [row.get(b, "") for b in backend_cols]

        for i in range(n):
            si = statuses[i]
            if not si:
                continue
            for j in range(n):
                sj = statuses[j]
                if not sj:
                    continue
                both_seen[i][j] += 1
                if si == "OK" and sj == "OK":
                    both_ok[i][j] += 1

    # convert to %
    data = []
    for i in range(n):
        r = []
        for j in range(n):
            d = both_seen[i][j]
            r.append((100.0 * both_ok[i][j] / d) if d else 0.0)
        data.append(r)

    fig_w = max(8, 0.6 * n)
    fig_h = max(7, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(backend_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(backend_cols, fontsize=8)

    ax.set_title("% Tests Where Both Backends Are OK")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Both-OK (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to results JSONL file")
    ap.add_argument("--out", default="report_support_matrix", help="Output directory (default: report_support_matrix)")
    args = ap.parse_args()

    in_path = args.jsonl
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    objs = list(read_jsonl(in_path))
    run_meta, item_ids, backend_cols, per_item, item_meta, raw_items_by_id = build_matrix(objs)

    csv_path = os.path.join(out_dir, "matrix_wide.csv")
    md_path  = os.path.join(out_dir, "report.md")
    bar_path = os.path.join(out_dir, "support_outcomes.png")
    overlap_path = os.path.join(out_dir, "backend_overlap.png")
    

    write_matrix_wide_csv(csv_path, item_ids, backend_cols, per_item, item_meta)
    write_report_md(md_path, run_meta, item_ids, backend_cols, per_item, item_meta, raw_items_by_id)
    make_outcome_stacked_bar_png(bar_path, item_ids, backend_cols, per_item)
    make_backend_overlap_png(overlap_path, item_ids, backend_cols, per_item)

    print("Wrote:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", bar_path)
    print(" -", overlap_path)


if __name__ == "__main__":
    main()
