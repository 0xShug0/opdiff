#!/usr/bin/env python3
# Latency report for JSONL:
# - CSV: op/id rows, backend median-ms columns (from res[].ms)
# - Figure1: CDF of R(op)=ms_backend/ms_baseline for each non-baseline backend
# - Figure2: For each backend, % of ops faster/slower than baseline (stacked bar)
# - MD: summary stats + counts
#
# Usage:
#   python report_latency.py path/to/results.jsonl
#   python report_latency.py path/to/results.jsonl --baseline torch_cpu_fp32
#   python report_latency.py path/to/results.jsonl --out report_latency
#
# Output dir (default): report_latency/
#   latency_wide.csv
#   latency_ratio_cdf.png
#   latency_faster_slower.png
#   report.md

import os
import sys
import json
import math
import csv
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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



def write_latency_wide_csv(path, item_ids, backend_list, per_item, item_op):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "op"] + backend_list)
        for item_id in item_ids:
            op = item_op.get(item_id, "")
            row = per_item.get(item_id, {})
            vals = []
            for b in backend_list:
                rec = row.get(b, {})
                ms = rec.get("ms")
                st = rec.get("status", "")
                if st == "OK" and ms is not None:
                    vals.append(f"{ms:.6f}")
                else:
                    vals.append("")
            w.writerow([item_id, op] + vals)

def geomean(xs):
    if not xs:
        return None
    s = 0.0
    n = 0
    for x in xs:
        if x <= 0 or not math.isfinite(x):
            continue
        s += math.log(x)
        n += 1
    if n == 0:
        return None
    return math.exp(s / n)


def percentile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = int(math.ceil(p * len(ys))) - 1
    k = max(0, min(k, len(ys) - 1))
    return ys[k]


def plot_ratio_cdf(path, ratios, baseline):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = list(plt.get_cmap("Dark2").colors)
    linestyles = ["-", "--", ":", "-."]

    plotted = 0
    keys = sorted(ratios.keys())

    for i, b in enumerate(keys):
        xs = ratios.get(b, [])
        if not xs:
            continue

        ys = sorted(xs)
        n = len(ys)
        cdf = [(j + 1) / n for j in range(n)]

        color = colors[i % len(colors)]
        linestyle = linestyles[(i // 10) % len(linestyles)] 

        ax.plot(ys, cdf, label=b, color=color, linestyle=linestyle)
        plotted += 1

    ax.set_title(f"CDF of Relative Latency R(op) vs baseline='{baseline}'")
    ax.set_xlabel("R(op) = ms_backend / ms_baseline  ( <1 faster, >1 slower )")
    ax.set_ylabel("CDF")
    ax.grid(True, linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(1.0, linewidth=1.0, linestyle="--")

    if plotted:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No comparable ratios to plot", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def plot_faster_slower(path, ratios, baseline):
    """
    For each backend (non-baseline), show % faster and % slower vs baseline.
    faster: R < 1
    slower: R > 1
    """
    backends = [b for b in sorted(ratios.keys())]
    faster_pct, slower_pct, counts = [], [], []

    for b in backends:
        xs = ratios.get(b, [])
        n = len(xs)
        counts.append(n)
        if n == 0:
            faster_pct.append(0.0)
            slower_pct.append(0.0)
            continue
        faster = sum(1 for x in xs if x < 1.0)
        slower = sum(1 for x in xs if x > 1.0)
        faster_pct.append(100.0 * faster / n)
        slower_pct.append(100.0 * slower / n)

    fig_w = max(9, 0.55 * len(backends))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    x = list(range(len(backends)))
    ax.bar(x, faster_pct, label="faster (R<1)")
    ax.bar(x, slower_pct, bottom=faster_pct, label="slower (R>1)")

    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Percent of comparable ops (%)")

    # put baseline on its own line to avoid smashing the top row
    ax.set_title("% Ops Faster/Slower (baseline = " + repr(baseline) + ")", pad=30)

    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linewidth=0.3)

    # --- key fix #1: legend OUTSIDE axes, in the figure margin ---
    # (top-right works well for wide categorical bars)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
        frameon=False,
        borderaxespad=0.0,
    )

    # --- key fix #2: N labels in AXES coordinates (not data coords) ---
    # y=1.01 means "just above the axes top", and clip_on=False keeps it visible
    for i, n in enumerate(counts):
        ax.text(
            i, 1.01, f"N={n}",
            transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
            ha="center", va="bottom", fontsize=8,
            clip_on=False,
        )

    # reserve right margin for the legend; reserve top margin for N/title
    fig.subplots_adjust(right=0.82, top=0.88)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report_md(path, run_meta, backend_list, baseline, ratios, counts):
    lines = []
    lines.append("# Latency Report (mode2)\n")
    lines.append(f"- Baseline for ratios: **{baseline}**")
    lines.append(f"- Total items: **{counts['total_items']}**")
    lines.append(f"- Items where baseline has OK+ms: **{counts['baseline_ok']}**")
    lines.append(f"- Items where baseline backend record is missing: **{counts['baseline_missing']}**\n")

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

    lines.append("## Backend OK counts (ms present)\n")
    lines.append("| Backend | OK count |")
    lines.append("|---|---:|")
    for b in backend_list:
        lines.append(f"| {b} | {counts['per_backend_ok'].get(b, 0)} |")
    lines.append("")

    lines.append("## Relative latency vs baseline (R = ms_backend / ms_baseline)\n")
    lines.append("Only includes ops where **both** backend and baseline are `OK` and have `ms`.\n")
    lines.append("| Backend | Comparable ops | GeoMean R | Median R | P90 R | % faster (R<1) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    if counts["baseline_ok"] > 0:
        lines.append(f"| {baseline} (baseline) | {counts['baseline_ok']} | 1.0000 | 1.0000 | 1.0000 | 0.0% |")
    else:
        lines.append(f"| {baseline} (baseline) | 0 | - | - | - | - |")

    for b in backend_list:
        if b == baseline:
            continue
        xs = ratios.get(b, [])
        if not xs:
            lines.append(f"| {b} | 0 | - | - | - | - |")
            continue
        gm = geomean(xs)
        med = median(xs)
        p90 = percentile(xs, 0.90)
        faster = sum(1 for x in xs if x < 1.0)
        pct_faster = 100.0 * faster / len(xs) if xs else 0.0
        lines.append(
            f"| {b} | {len(xs)} | {gm:.4f} | {med:.4f} | {p90:.4f} | {pct_faster:.1f}% |"
        )
    lines.append("")

    lines.append("## Figures\n")
    lines.append("- `latency_ratio_cdf.png`: CDF of per-op ratios R(op) vs baseline.\n")
    lines.append("- `latency_faster_slower.png`: fraction of comparable ops that are faster/slower than baseline.\n")

    lines.append("## Notes\n")
    lines.append("- The baseline here is **only** for computing R(op); it does not need to match any correctness-diff baseline.\n")
    lines.append("- If baseline backend name is wrong/missing, use `--baseline <name>`.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def extract_items(objs):
    run_meta = None
    all_backends = set()
    per_item = {}
    item_op = {}

    def _parse_times_all(times_all):
        if not (isinstance(times_all, list) and times_all):
            return None
        last = times_all[-1]
        if not isinstance(last, list):
            return None
        ts = []
        for x in last[1:]:
            v = _to_float(x)
            if v is not None and math.isfinite(v) and v > 0:
                ts.append(v)
        return ts or None

    for obj in objs:
        if is_meta(obj) and run_meta is None:
            run_meta = obj.get("meta")

        if not is_item(obj):
            continue

        item_id = str(obj.get("id", "") or "")
        if not item_id:
            continue

        op_name = ""
        cases = obj.get("cases") or []
        if isinstance(cases, list) and cases and isinstance(cases[0], dict):
            op_name = str(cases[0].get("op", "") or "")
        item_op[item_id] = op_name

        row = {}

        base = obj.get("baseline")
        if isinstance(base, dict):
            base_backend = str(base.get("backend", "") or "")
            if base_backend:
                all_backends.add(base_backend)
                base_status = str(base.get("status", "") or "")
                base_ms = _to_float(base.get("ms"))
                base_times = _parse_times_all(base.get("times_all"))
                row[base_backend] = {
                    "status": base_status,
                    "ms": base_ms,
                    "times": base_times,
                }

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

            res = b.get("res") or []
            if not isinstance(res, list):
                res = []

            statuses = []
            ms_vals = []
            for r in res:
                if not isinstance(r, dict):
                    continue
                statuses.append(str(r.get("status", "") or ""))
                ms = _to_float(r.get("ms"))
                if ms is not None:
                    ms_vals.append(ms)

            if statuses:
                combined_status = "OK"
                for st in statuses:
                    if st != "OK":
                        combined_status = st
                        break
            else:
                combined_status = ""

            times = _parse_times_all(b.get("times_all"))

            new_rec = {
                "status": combined_status,
                "ms": median(ms_vals),
                "times": times,
            }

            old = row.get(backend)
            if old is None:
                row[backend] = new_rec
            else:
                old_ok = (old.get("status") == "OK" and old.get("ms") is not None and old.get("ms") > 0)
                new_ok = (new_rec.get("status") == "OK" and new_rec.get("ms") is not None and new_rec.get("ms") > 0)
                if new_ok or not old_ok:
                    row[backend] = new_rec
                else:
                    if old.get("times") is None and new_rec.get("times") is not None:
                        old["times"] = new_rec["times"]

        per_item[item_id] = row

    item_ids = sorted(per_item.keys())
    backend_list = sorted(all_backends)
    return run_meta, item_ids, backend_list, per_item, item_op



def compute_ratios(item_ids, backend_list, per_item, baseline, per_run=False):
    ratios = {b: [] for b in backend_list if b != baseline}
    counts = {
        "total_items": len(item_ids),
        "baseline_ok": 0,
        "baseline_missing": 0,
        "per_backend_ok": Counter(),
        "per_backend_comparable": Counter(),
    }

    for item_id in item_ids:
        row = per_item.get(item_id, {})
        base_rec = row.get(baseline)
        if not base_rec:
            counts["baseline_missing"] += 1
            continue

        base_ok = (
            base_rec.get("status") == "OK"
            and base_rec.get("ms") is not None
            and base_rec.get("ms") > 0
        )
        if base_ok:
            counts["baseline_ok"] += 1

        base_times = base_rec.get("times")
        base_times_ok = isinstance(base_times, list) and len(base_times) > 0

        for b in backend_list:
            if b == baseline:
                continue
            rec = row.get(b)
            if not rec:
                continue

            ok = (
                rec.get("status") == "OK"
                and rec.get("ms") is not None
                and rec.get("ms") > 0
            )
            if ok:
                counts["per_backend_ok"][b] += 1
            if base_ok:
                counts["per_backend_ok"][baseline] += 1

            if not (base_ok and ok):
                continue

            if per_run:
                xs = rec.get("times")
                if isinstance(xs, list) and xs and base_times_ok:
                    n = min(len(xs), len(base_times))
                    for i in range(n):
                        r = xs[i] / base_times[i]
                        if r > 0 and math.isfinite(r):
                            ratios[b].append(r)
                            counts["per_backend_comparable"][b] += 1
                else:
                    r = rec["ms"] / base_rec["ms"]
                    if r > 0 and math.isfinite(r):
                        ratios[b].append(r)
                        counts["per_backend_comparable"][b] += 1
            else:
                r = rec["ms"] / base_rec["ms"]
                if r > 0 and math.isfinite(r):
                    ratios[b].append(r)
                    counts["per_backend_comparable"][b] += 1

    return ratios, counts



def _apply_test_id_filter(item_ids, wanted_ids):
    if not wanted_ids:
        return item_ids
    wanted = set(wanted_ids)
    return [x for x in item_ids if x in wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to results JSONL file")
    ap.add_argument("--out", default="report_latency", help="Output directory (default: report_latency)")
    ap.add_argument("--baseline", default="torch_cpu_fp32", help="Baseline backend name (default: torch_cpu_fp32)")
    ap.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="Filter to specific testcase id (repeatable). Example: --test-id foo --test-id bar",
    )
    args = ap.parse_args()

    out_dir = args.out
    baseline = args.baseline
    os.makedirs(out_dir, exist_ok=True)

    objs = list(read_jsonl(args.jsonl))
    run_meta, item_ids, backend_list, per_item, item_op = extract_items(objs)

    if baseline not in backend_list:
        msg = []
        msg.append(f"Baseline backend '{baseline}' not found in file.")
        msg.append("Available backends:")
        for b in backend_list:
            msg.append(f"  - {b}")
        raise SystemExit("\n".join(msg))

    if args.test_id:
        item_ids2 = _apply_test_id_filter(item_ids, args.test_id)
        if not item_ids2:
            print("No items matched --test-id filter:", ", ".join(args.test_id), file=sys.stderr)
            sys.exit(2)
        item_ids = item_ids2

    csv_path = os.path.join(out_dir, "latency_wide.csv")
    cdf_path = os.path.join(out_dir, "latency_ratio_cdf.png")
    fs_path = os.path.join(out_dir, "latency_faster_slower.png")
    md_path = os.path.join(out_dir, "report.md")

    write_latency_wide_csv(csv_path, item_ids, backend_list, per_item, item_op)

    per_run = bool(args.test_id)
    ratios, counts = compute_ratios(item_ids, backend_list, per_item, baseline, per_run=per_run)
    plot_ratio_cdf(cdf_path, ratios, baseline)
    plot_faster_slower(fs_path, ratios, baseline)
    write_report_md(md_path, run_meta, backend_list, baseline, ratios, counts)

    print("Wrote:")
    print(" -", csv_path)
    print(" -", cdf_path)
    print(" -", fs_path)
    print(" -", md_path)



if __name__ == "__main__":
    main()
