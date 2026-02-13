#!/usr/bin/env python3
# Drop-in report script for output differences (max_abs) from the JSONL format produced by jsonl_writer/core.
#
# Generates (mirrors the style of report_latency + report_support_matrix):
#   - diff_max_abs_cdf.png : CDF of max_abs
#       * default: per-item max_abs (median over repeats in diff_all)
#       * if --test-id is provided: per-run max_abs points from diff_all (each repeat is a sample)
#   - diff_buckets.png     : stacked bar per backend across fixed buckets + missing/error
#       * default: per-item buckets (median over repeats)
#       * if --test-id is provided: buckets computed over "observations" where OK repeats count as samples,
#                    and ERR/MISSING count once per item
#   - output_diff_wide.csv : id/op rows, backend cols = per-item max_abs (median), blank if missing/error
#   - report.md            : counts + summary stats (per-item median semantics)
#
# Usage:
#   python report_output_diff.py path/to/results.jsonl
#   python report_output_diff.py path/to/results.jsonl --out report_output_diff
#   python report_output_diff.py path/to/results.jsonl --test-id some_id
#   python report_output_diff.py path/to/results.jsonl --test-id id1 --test-id id2

import os
import sys
import json
import math
import csv
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed buckets
_BUCKETS = [
    ("==0", lambda v: v == 0.0),
    ("(0,1e-6]", lambda v: 0.0 < v <= 1e-6),
    ("(1e-6,1e-4]", lambda v: 1e-6 < v <= 1e-4),
    ("(1e-4,1e-2]", lambda v: 1e-4 < v <= 1e-2),
    ("(1e-2,1e-1]", lambda v: 1e-2 < v <= 1e-1),
    ("(1e-1,1]", lambda v: 1e-1 < v <= 1.0),
    (">1", lambda v: v > 1.0),
]
_BUCKET_NAMES = [b[0] for b in _BUCKETS] + ["DIFF_ERR", "MISSING"]


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


def is_meta(obj):
    return isinstance(obj, dict) and obj.get("type") == "meta"


def is_item(obj):
    return isinstance(obj, dict) and obj.get("type") == "item"


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        t = x.strip()
        if not t or t.lower() == "x":
            return None
        try:
            return float(t)
        except Exception:
            return None
    return None


def median(vals):
    if not vals:
        return None
    xs = sorted(vals)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def percentile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = int(math.ceil(p * len(ys))) - 1
    k = max(0, min(k, len(ys) - 1))
    return ys[k]


def combine_status(res_list):
    """Collapse res[].status to a single status: 'OK' if all OK, else first non-OK."""
    if not isinstance(res_list, list) or not res_list:
        return ""
    statuses = []
    for r in res_list:
        if isinstance(r, dict):
            statuses.append(str(r.get("status", "") or ""))
    if not statuses:
        return ""
    for st in statuses:
        if st != "OK":
            return st
    return "OK"


def summarize_diff_all(diff_all):
    """
    Returns (diff_state, med_max_abs, run_max_abs_list, err_code, err_msg)
      diff_state in {'OK','ERR','MISSING'}

    - diff_all is None or empty -> MISSING
    - diff_all has entries:
        * if any entry ok=False -> ERR (use its code/msg)
        * else collect max_abs from ok=True entries:
            - run_max_abs_list = all repeats' max_abs
            - med_max_abs = median(run_max_abs_list)
    """
    if diff_all is None:
        return "MISSING", None, [], None, None
    if not isinstance(diff_all, list) or not diff_all:
        return "MISSING", None, [], None, None

    run_vals = []
    for d in diff_all:
        if not isinstance(d, dict):
            return "ERR", None, [], "BAD_RECORD", "diff_all entry is not an object"
        ok = d.get("ok")
        if ok is True:
            v = _to_float(d.get("max_abs"))
            if v is None or (not math.isfinite(v)) or v < 0:
                return "ERR", None, [], "NONFINITE", f"max_abs not finite: {d.get('max_abs')!r}"
            run_vals.append(v)
        else:
            code = str(d.get("code", "DIFF_ERR") or "DIFF_ERR")
            msg = str(d.get("msg", "") or "")
            return "ERR", None, [], code, msg

    if not run_vals:
        return "MISSING", None, [], None, None
    return "OK", median(run_vals), run_vals, None, None


def extract_items(objs):
    run_meta = None
    all_backends = set()
    per_item = {}
    item_op = {}

    baseline_backends_seen = set()

    for obj in objs:
        if is_meta(obj) and run_meta is None:
            run_meta = obj.get("meta")

        if not is_item(obj):
            continue

        item_id = str(obj.get("id", "") or "")
        if not item_id:
            continue

        base = obj.get("baseline")
        if isinstance(base, dict):
            bb = str(base.get("backend", "") or "")
            if bb:
                baseline_backends_seen.add(bb)

        op_name = ""
        cases = obj.get("cases") or []
        if isinstance(cases, list) and cases and isinstance(cases[0], dict):
            op_name = str(cases[0].get("op", "") or "")
        item_op[item_id] = op_name

        row = {}
        backends = obj.get("backends") or []
        if not isinstance(backends, list):
            backends = []

        for b in backends:
            if not isinstance(b, dict):
                continue
            backend = str(b.get("backend", "") or "")
            if not backend:
                continue
            all_backends.add(backend)

            run_status = combine_status(b.get("res") or [])
            diff_state, med_max_abs, run_max_abs_list, err_code, err_msg = summarize_diff_all(
                b.get("diff_all", None)
            )

            row[backend] = {
                "run_status": run_status,
                "diff_state": diff_state,
                "max_abs": med_max_abs,            # per-item median
                "max_abs_runs": run_max_abs_list,  # per-run samples
                "diff_err_code": err_code,
                "diff_err_msg": err_msg,
            }

        per_item[item_id] = row

    item_ids = sorted(per_item.keys())
    backend_list = sorted(all_backends)

    if len(baseline_backends_seen) == 1:
        diff_baseline_backend = next(iter(baseline_backends_seen))
    elif len(baseline_backends_seen) == 0:
        diff_baseline_backend = ""
    else:
        diff_baseline_backend = "MULTIPLE: " + ", ".join(sorted(baseline_backends_seen))

    return run_meta, item_ids, backend_list, per_item, item_op, diff_baseline_backend


def write_diff_wide_csv(path, item_ids, backend_list, per_item, item_op):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "op"] + backend_list)
        for item_id in item_ids:
            op = item_op.get(item_id, "")
            row = per_item.get(item_id, {})
            vals = []
            for b in backend_list:
                rec = row.get(b, {})
                if rec.get("diff_state") == "OK" and rec.get("max_abs") is not None:
                    vals.append(f"{rec['max_abs']:.6g}")
                else:
                    vals.append("")
            w.writerow([item_id, op] + vals)


def compute_distributions_item_median(item_ids, backend_list, per_item):
    """
    Per-item distributions (medians) for report.md + CSV semantics.
    """
    diffs = {b: [] for b in backend_list}
    counts = {
        "total_items": len(item_ids),
        "per_backend_present": Counter(),
        "per_backend_run_ok": Counter(),
        "per_backend_diff_ok": Counter(),
        "per_backend_diff_err": Counter(),
        "per_backend_diff_missing": Counter(),
        "per_backend_diff_err_codes": defaultdict(Counter),
    }

    for item_id in item_ids:
        row = per_item.get(item_id, {})
        for b in backend_list:
            rec = row.get(b)
            if not rec:
                continue
            counts["per_backend_present"][b] += 1
            if rec.get("run_status") == "OK":
                counts["per_backend_run_ok"][b] += 1

            st = rec.get("diff_state")
            if st == "OK":
                counts["per_backend_diff_ok"][b] += 1
                v = rec.get("max_abs")
                if v is not None and v >= 0 and math.isfinite(v):
                    diffs[b].append(v)
            elif st == "ERR":
                counts["per_backend_diff_err"][b] += 1
                code = rec.get("diff_err_code") or "DIFF_ERR"
                counts["per_backend_diff_err_codes"][b][code] += 1
            else:
                counts["per_backend_diff_missing"][b] += 1

    return diffs, counts


def compute_plot_diffs(item_ids, backend_list, per_item, per_run):
    """
    Returns diffs_for_plot[b] = list of x values to use in CDF.

    - per_run=False: per-item median (same as report)
    - per_run=True: each ok repeat's max_abs from diff_all is a sample
    """
    diffs = {b: [] for b in backend_list}

    for item_id in item_ids:
        row = per_item.get(item_id, {})
        for b in backend_list:
            rec = row.get(b)
            if not rec:
                continue
            if rec.get("diff_state") != "OK":
                continue

            if per_run:
                xs = rec.get("max_abs_runs") or []
                for v in xs:
                    if v is not None and v >= 0 and math.isfinite(v):
                        diffs[b].append(v)
            else:
                v = rec.get("max_abs")
                if v is not None and v >= 0 and math.isfinite(v):
                    diffs[b].append(v)

    return diffs


def plot_diff_cdf(path, diffs, per_run):
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    colors = list(plt.get_cmap("Dark2").colors)
    linestyles = ["-", "--", ":", "-."]

    # Keep legend/line order stable
    keys = sorted(diffs.keys())

    for i, b in enumerate(keys):
        xs = diffs.get(b, [])
        if not xs:
            continue
        ys = sorted(xs)
        n = len(ys)
        cdf = [(j + 1) / n for j in range(n)]

        color = colors[i % len(colors)]
        linestyle = linestyles[(i // 10) % len(linestyles)]

        ax.plot(ys, cdf, label=b, color=color, linestyle=linestyle)
        plotted += 1

    ax.set_title("CDF of Output Difference Magnitude (max_abs)")
    ax.set_xlabel("max_abs (per-run)" if per_run else "max_abs (per-item median over diff_all repeats)")
    ax.set_ylabel("CDF")
    ax.grid(True, linewidth=0.3)
    ax.set_xscale("log")

    if plotted:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No diff values to plot", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bucket_name(v):
    if v is None or (not math.isfinite(v)) or v < 0:
        return None
    for name, pred in _BUCKETS:
        if pred(v):
            return name
    return None


def compute_bucket_percentages_item_median(item_ids, backend_list, per_item, counts):
    """
    Per-item bucket semantics (% of present items). Matches original behavior.
    """
    per_backend_bucket_pct = {b: Counter() for b in backend_list}

    for b in backend_list:
        denom = counts["per_backend_present"].get(b, 0)
        if denom <= 0:
            continue

        c = Counter()
        for item_id in item_ids:
            rec = per_item.get(item_id, {}).get(b)
            if not rec:
                continue
            st = rec.get("diff_state")
            if st == "OK":
                bn = bucket_name(rec.get("max_abs")) or "MISSING"
                c[bn] += 1
            elif st == "ERR":
                c["DIFF_ERR"] += 1
            else:
                c["MISSING"] += 1

        for bn in _BUCKET_NAMES:
            per_backend_bucket_pct[b][bn] = 100.0 * c.get(bn, 0) / denom

    return per_backend_bucket_pct


def compute_bucket_percentages_per_run(item_ids, backend_list, per_item):
    """
    Per-run bucket semantics for the *figure*:
      - Each OK repeat counts as one observation in its bucket.
      - Each ERR item counts once as DIFF_ERR.
      - Each MISSING item counts once as MISSING.
    Percentages are over total observations (OK repeats + ERR items + MISSING items).
    """
    per_backend_bucket_pct = {b: Counter() for b in backend_list}

    for b in backend_list:
        c = Counter()
        denom = 0

        for item_id in item_ids:
            rec = per_item.get(item_id, {}).get(b)
            if not rec:
                continue
            st = rec.get("diff_state")
            if st == "OK":
                xs = rec.get("max_abs_runs") or []
                for v in xs:
                    bn = bucket_name(v) or "MISSING"
                    c[bn] += 1
                    denom += 1
            elif st == "ERR":
                c["DIFF_ERR"] += 1
                denom += 1
            else:
                c["MISSING"] += 1
                denom += 1

        if denom <= 0:
            continue

        for bn in _BUCKET_NAMES:
            per_backend_bucket_pct[b][bn] = 100.0 * c.get(bn, 0) / denom

    return per_backend_bucket_pct


def plot_diff_buckets(path, backend_list, per_backend_bucket_pct, per_run):
    fig, ax = plt.subplots(figsize=(10, 6))

    xs = list(range(len(backend_list)))
    bottoms = [0.0] * len(backend_list)

    for bn in _BUCKET_NAMES:
        heights = [per_backend_bucket_pct.get(b, {}).get(bn, 0.0) for b in backend_list]
        ax.bar(xs, heights, bottom=bottoms, label=bn)
        bottoms = [bottoms[i] + heights[i] for i in range(len(bottoms))]

    ax.set_title("Output Difference Buckets (max_abs) per Backend" + (" (per-run)" if per_run else ""))
    ax.set_xlabel("Backend")
    ax.set_ylabel("Percent of observations" if per_run else "Percent of items (present)")
    ax.set_xticks(xs)
    ax.set_xticklabels(backend_list, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linewidth=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report_md(path, run_meta, backend_list, diffs, counts, per_backend_bucket_pct, diff_baseline_backend):
    lines = []
    lines.append("# Output Difference Report (max_abs)\n")
    lines.append(f"- Total items: **{counts['total_items']}**\n")

    if diff_baseline_backend:
        lines.append(f"- Diff baseline backend (from JSONL `item.baseline.backend`): **{diff_baseline_backend}**\n")
        lines.append("> Note: `diff_all` is already computed vs this baseline at run time; this report does not rebaseline post-hoc.\n")
    else:
        lines.append("- Diff baseline backend: _(not found in JSONL)_\n")

    if isinstance(run_meta, dict):
        lines.append("## Run metadata (best-effort)\n")
        try:
            txt = json.dumps(run_meta, indent=2, ensure_ascii=False)
            if len(txt) > 2000:
                txt = txt[:2000] + "\n... (truncated)\n"
            lines.append("```json")
            lines.append(txt)
            lines.append("```\n")
        except Exception:
            lines.append("_Failed to render run metadata as JSON._\n")

    lines.append("## Backend counts\n")
    lines.append("| Backend | Present | Run OK | Diff OK | Diff ERR | Diff missing |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in backend_list:
        lines.append(
            f"| {b} | {counts['per_backend_present'].get(b, 0)}"
            f" | {counts['per_backend_run_ok'].get(b, 0)}"
            f" | {counts['per_backend_diff_ok'].get(b, 0)}"
            f" | {counts['per_backend_diff_err'].get(b, 0)}"
            f" | {counts['per_backend_diff_missing'].get(b, 0)} |"
        )
    lines.append("")

    lines.append("## max_abs summary (only Diff OK)\n")
    lines.append("| Backend | N | Median | P90 | P99 | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in backend_list:
        xs = diffs.get(b, [])
        if not xs:
            lines.append(f"| {b} | 0 | - | - | - | - |")
            continue
        med = median(xs)
        p90 = percentile(xs, 0.90)
        p99 = percentile(xs, 0.99)
        mx = max(xs)
        lines.append(f"| {b} | {len(xs)} | {med:.6g} | {p90:.6g} | {p99:.6g} | {mx:.6g} |")
    lines.append("")

    lines.append("## Bucket breakdown (% of present items)\n")
    lines.append("| Backend | " + " | ".join(_BUCKET_NAMES) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(_BUCKET_NAMES)) + "|")
    for b in backend_list:
        parts = []
        for bn in _BUCKET_NAMES:
            v = per_backend_bucket_pct.get(b, {}).get(bn, 0.0)
            parts.append(f"{v:.1f}%")
        lines.append("| " + b + " | " + " | ".join(parts) + " |")
    lines.append("")

    lines.append("## Top diff error codes (if any)\n")
    for b in backend_list:
        c = counts["per_backend_diff_err_codes"].get(b)
        if not c:
            continue
        common = c.most_common(8)
        lines.append(f"### {b}\n")
        lines.append("| Code | Count |")
        lines.append("|---|---:|")
        for code, n in common:
            lines.append(f"| {code} | {n} |")
        lines.append("")

    lines.append("## Figures\n")
    lines.append("- `diff_max_abs_cdf.png`: CDF of `max_abs` (default per-item median; if `--test-id` is provided, uses each repeat).\n")
    lines.append("- `diff_buckets.png`: stacked bucket distribution per backend (default per-item; if `--test-id` is provided, uses repeat-level observations + item-level ERR/MISSING).\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _apply_test_id_filter(item_ids, item_op, wanted_ids):
    if not wanted_ids:
        return item_ids
    wanted = set(wanted_ids)
    out = [x for x in item_ids if x in wanted]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to results JSONL file")
    ap.add_argument("--out", default="report_output_diff", help="Output directory (default: report_output_diff)")
    ap.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="Filter to specific testcase id (repeatable). Example: --test-id foo --test-id bar",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    objs = list(read_jsonl(args.jsonl))
    run_meta, item_ids, backend_list, per_item, item_op, diff_baseline_backend = extract_items(objs)

    # Apply testcase-id filter (if requested)
    if args.test_id:
        item_ids2 = _apply_test_id_filter(item_ids, item_op, args.test_id)
        if not item_ids2:
            print("No items matched --test-id filter:", ", ".join(args.test_id), file=sys.stderr)
            sys.exit(2)
        item_ids = item_ids2

    # If --test-id is provided, figures should always use per-run samples.
    per_run = bool(args.test_id)

    csv_path = os.path.join(args.out, "output_diff_wide.csv")
    cdf_path = os.path.join(args.out, "diff_max_abs_cdf.png")
    buck_path = os.path.join(args.out, "diff_buckets.png")
    md_path = os.path.join(args.out, "report.md")

    # CSV remains per-item median semantics
    write_diff_wide_csv(csv_path, item_ids, backend_list, per_item, item_op)

    # Report/MD counts + stats remain per-item median semantics
    diffs_item, counts = compute_distributions_item_median(item_ids, backend_list, per_item)

    # Figures: per-run iff --test-id is provided
    diffs_for_plot = compute_plot_diffs(item_ids, backend_list, per_item, per_run=per_run)
    plot_diff_cdf(cdf_path, diffs_for_plot, per_run=per_run)

    if per_run:
        per_backend_bucket_pct_plot = compute_bucket_percentages_per_run(item_ids, backend_list, per_item)
    else:
        per_backend_bucket_pct_plot = compute_bucket_percentages_item_median(item_ids, backend_list, per_item, counts)

    plot_diff_buckets(buck_path, backend_list, per_backend_bucket_pct_plot, per_run=per_run)

    # MD bucket table stays item-median semantics (so it matches counts)
    per_backend_bucket_pct_md = compute_bucket_percentages_item_median(item_ids, backend_list, per_item, counts)
    write_report_md(md_path, run_meta, backend_list, diffs_item, counts, per_backend_bucket_pct_md, diff_baseline_backend)

    print("Wrote:")
    print(" -", csv_path)
    print(" -", cdf_path)
    print(" -", buck_path)
    print(" -", md_path)


if __name__ == "__main__":
    main()
