"""Numeric diff utilities for comparing backend outputs against a PyTorch baseline.

This module provides:
- `_flatten_out`: deterministic flattening of nested output structures into a tuple
  (with stable dict key ordering).
- `diff_metrics`: structural compatibility checks plus basic error metrics (MSE, max abs,
  mean abs) with explicit failure codes for NaNs and shape/emptiness mismatches.

All comparisons are performed in float64 on CPU numpy arrays for stability.
"""
import numbers
import torch
import numpy as np

def _flatten_out(x):
    """Flatten nested outputs (dict/list/tuple) into a tuple with deterministic ordering.

    Dicts are flattened by sorted key order to ensure stable alignment across runs/backends.
    Non-container leaves are returned as a one-element tuple.
    """
    if isinstance(x, dict):
        out = []
        for k in sorted(x.keys()):
            out.extend(_flatten_out(x[k]))
        return tuple(out)
    if isinstance(x, (tuple, list)):
        out = []
        for xi in x:
            out.extend(_flatten_out(xi))
        return tuple(out)
    return (x,)


def diff_metrics(baseline_out, backend_out):
    """Compute elementwise error metrics between baseline and backend outputs.

    This function compares nested output structures (tensors, arrays, scalars, and
    containers) and returns aggregate error metrics when the structures are comparable.

    If outputs are containers, they are flattened into a linear tuple in a deterministic
    way (dicts by sorted key order) so corresponding leaves can be compared.

    Returns explicit failure codes when outputs are not comparable (shape/type mismatch,
    empty outputs) or contain NaNs.

    Notes:
        - Values are normalized to CPU numpy arrays and compared in float64.
        - Rank-0 scalars are treated as shape-(1,) arrays to simplify comparisons.
    """

    if isinstance(baseline_out, (tuple, list, dict)) or isinstance(backend_out, (tuple, list, dict)):
        baseline_out = _flatten_out(baseline_out)
        backend_out = _flatten_out(backend_out)

    def norm(x):
        """Normalize a leaf value to a numpy ndarray on CPU (rank-0 -> shape (1,))."""
        if torch.is_tensor(x):
            if x.is_conj():
                x = x.resolve_conj()
            x = x.detach().cpu().numpy()
        elif isinstance(x, (bool, numbers.Number, np.generic)):
            x = np.asarray(x)
        x = np.asarray(x)
        return x.reshape((1,)) if x.shape == () else x

    def has_nan(x: np.ndarray) -> bool:
        """Return True if array contains any NaN."""
        return bool(np.isnan(x).any())

    mse_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    n = 0

    def walk(a, b):
        """Recursively validate structure and accumulate metrics over leaf pairs."""
        nonlocal mse_sum, abs_sum, max_abs, n
        # Sequence alignment.
        if isinstance(a, (tuple, list)):
            if not isinstance(b, (tuple, list)):
                # Permit a single-element wrapper vs scalar leaf.
                if len(a) == 1:
                    return walk(a[0], b)
                return False, "NOT_COMPARABLE_SHAPE", "type mismatch"
            if len(a) != len(b):
                return False, "NOT_COMPARABLE_SHAPE", "len mismatch"
            for ai, bi in zip(a, b):
                ok, code, msg = walk(ai, bi)
                if not ok:
                    return ok, code, msg
            return True, "NONE", None
        
        # # Dict alignment by key set and sorted traversal for determinism.
        # if isinstance(a, dict):
        #     if not isinstance(b, dict) or set(a) != set(b):
        #         return False, "NOT_COMPARABLE_SHAPE", "key/type mismatch"
        #     for k in sorted(a):
        #         ok, code, msg = walk(a[k], b[k])
        #         if not ok:
        #             return ok, code, msg
        #     return True, "NONE", None
        
        # Leaf comparison.
        ta, tb = norm(a), norm(b)
        if ta.shape != tb.shape:
            return False, "NOT_COMPARABLE_SHAPE", "shape mismatch"
        
        # Treat empty outputs as non-comparable (avoids dividing by zero / misleading metrics).
        if ta.size == 0 and tb.size == 0:
            return False, "NOT_COMPARABLE_EMPTY", None
        if ta.size == 0 or tb.size == 0:
            return False, "NOT_COMPARABLE_EMPTY", None
        
        # NaNs are surfaced explicitly as failure codes.
        if has_nan(ta):
            return False, "NAN_IN_BASELINE", "baseline contains NaN"
        if has_nan(tb):
            return False, "NAN_IN_OUTPUT", "output contains NaN"

        # Accumulate in float64 for numeric stability.
        da = ta.astype(np.float64, copy=False)
        db = tb.astype(np.float64, copy=False)
        d = da - db
        ad = np.abs(d)

        max_abs = max(max_abs, float(np.max(ad)))
        abs_sum += float(np.sum(ad))
        mse_sum += float(np.sum(d * d))
        n += int(d.size)
        return True, "NONE", None

    ok, code, msg = walk(baseline_out, backend_out)
    if not ok or n == 0:
        return None, None, None, code, msg

    return mse_sum / n, max_abs, abs_sum / n, "NONE", None