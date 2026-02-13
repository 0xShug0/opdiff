"""
YAML config loading with include/merge and template expansion.

This module loads YAML test configurations and returns a fully resolved `Config`
ready for execution.

It supports:
- Recursive `include:` directives (paths resolved relative to the including file)
  with cycle detection.
- Deterministic merge semantics:
  - `dims` and `presets` are strict-merged (conflicting values raise).
  - `tests` are concatenated (included tests first, then local tests).
  - All other keys are overridden by the current file.
- Shorthand node normalization:
  - `{ref: ...}` and `{var: ...}` are normalized into explicit typed node forms
    before schema validation/parsing.
- Template expansion:
  - `TemplateModuleNode` expands into concrete `ModuleNode` tests by taking the
    cartesian product of template variables and substituting them into nodes.
  - `TemplateComparePairNode` expands into a `PairTest` that wraps two concrete
    module tests (`__a` and `__b`) generated from shared variables/cases.
  - Test ids get a stable, filesystem-friendly suffix of the form:
      __k=v__k=v
    where values are sanitized to `[A-Za-z0-9_-]`.

Determinism:
- Variable keys are sorted when constructing id suffixes, ensuring stable ids
  and expansion order across runs given the same inputs.

Entrypoints:
- `load_config(path)`: load from YAML file, resolve includes, validate, expand templates.
- `load_config_from_obj(raw_dict)`: same pipeline starting from an in-memory dict.
"""

import itertools
import os
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple

from testgen.test_plan import Config, PairTest, RefNode
from testgen.test_validator import ConstructNode, ModuleNode, TemplateComparePairNode, TemplateModuleNode, TestItem, VarNode, parse_node


def _normalize_refs(obj: Any) -> Any:
    """Normalize YAML shorthand forms like {ref: ...} / {var: ...} into explicit typed nodes recursively."""
    if isinstance(obj, dict):
        # {ref: ...} shorthand (and normalize nested value too)
        if "ref" in obj and "type" not in obj and len(obj) == 1:
            return {"type": "ref", "ref": _normalize_refs(obj["ref"])}

        # {var: ...} shorthand (and normalize nested value too)
        if "var" in obj and "type" not in obj and len(obj) == 1:
            return {"type": "var", "name": _normalize_refs(obj["var"])}

        return {k: _normalize_refs(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_normalize_refs(v) for v in obj]

    return obj

def _as_list(x: Any) -> List[str]:
    """Coerce a value into a list of strings (None → [], str → [str], list[str] → list[str])."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list) and all(isinstance(i, str) for i in x):
        return x
    raise ValueError("include must be a string or a list of strings")


def _merge_dicts_strict(a: Dict[str, Any], b: Dict[str, Any], *, where: str) -> Dict[str, Any]:
    """Merge two dicts; Raise if the same key has conflicting values."""
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and out[k] != v:
            raise ValueError(f"conflict in {where}.{k}: {out[k]!r} vs {v!r}")
        out[k] = v
    return out


def _merge_configs(base: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """Merge included and current configs: strict-merge dims/presets, concat tests, override other keys."""
    out: Dict[str, Any] = {}
    out["dims"] = _merge_dicts_strict(base.get("dims", {}), cur.get("dims", {}), where="dims")
    out["presets"] = _merge_dicts_strict(
        base.get("presets", {}), cur.get("presets", {}), where="presets"
    )
    out["tests"] = (base.get("tests", []) or []) + (cur.get("tests", []) or [])

    for k, v in cur.items():
        if k in {"include", "dims", "presets", "tests"}:
            continue
        out[k] = v
    return out


def _load_raw_with_includes(path: str, *, _stack: Set[str]) -> Dict[str, Any]:
    """Load a YAML config, recursively resolve includes, and detect include cycles."""
    apath = os.path.abspath(path)
    if apath in _stack:
        chain = " -> ".join(list(_stack) + [apath])
        raise ValueError(f"include cycle detected: {chain}")

    _stack.add(apath)
    with open(apath, "r", encoding="utf-8") as f:
        cur = yaml.safe_load(f) or {}
        if not isinstance(cur, dict):
            raise ValueError(f"top-level YAML must be a mapping/dict: {path}")

    includes = _as_list(cur.get("include"))
    merged: Dict[str, Any] = {"dims": {}, "presets": {}, "tests": []}

    base_dir = os.path.dirname(apath)
    for inc in includes:
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        inc_raw = _load_raw_with_includes(inc_path, _stack=_stack)
        merged = _merge_configs(merged, inc_raw)

    merged = _merge_configs(merged, cur)
    _stack.remove(apath)
    return merged


############## test case expand ###############
def _safe_id_part(v: Any) -> str:
    """Convert a value to an ID-safe string by replacing non [A-Za-z0-9_-] characters with '_'."""
    s = str(v)
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s)

def _id_suffix_format(keys_all: List[str], env: Dict[str, Any]) -> str:
    """
    Build the variable suffix portion of a test id in a stable, readable schema.

    Schema:
      - Test ids are built as: <base_id><suffix>
      - suffix is either:
          ""                                   (when there are no template vars)
        or
          "__k=v__k=v__k=v"                    (when template vars exist)

    Rules:
      - "__" is the segment separator between key/value segments.
      - Each segment is "k=v" (NOT "k_v") to avoid ambiguity when values contain underscores.
      - Values are sanitized via _safe_id_part() so the id stays filesystem/test-runner friendly.
      - keys_all should be deterministic (you already sort it upstream), so ids are stable across runs.
    """
    parts = [f"{k}={_safe_id_part(env[k])}" for k in keys_all]
    return ("__" + "__".join(parts)) if parts else ""


def _subst_vars_in_any(x: Any, env: Dict[str, Any]) -> Any:
    """Substitute template variables inside nodes, lists, and dicts using the provided config."""
    if isinstance(x, VarNode):
        if x.name not in env:
            raise KeyError(f"Unknown template var '{x.name}'")
        v = env[x.name]
        if isinstance(v, str) and v in env and isinstance(env[v], int):
            return env[v]
        return v

    if isinstance(x, RefNode):
        if isinstance(x.ref, VarNode):
            name = x.ref.name
            if name not in env:
                raise KeyError(f"Unknown template var '{name}'")
            v = env[name]
            if not isinstance(v, str):
                raise ValueError(
                    f"Ref var '{name}' must expand to a preset name string, got {type(v)}"
                )
            return RefNode(type="ref", ref=v)
        return x

    if isinstance(x, ConstructNode):
        return ConstructNode(
            type="construct",
            path=x.path,
            args=[_subst_vars_in_any(a, env) for a in (x.args or [])],
            kwargs={k: _subst_vars_in_any(v, env) for k, v in (x.kwargs or {}).items()},
        )

    if isinstance(x, ModuleNode):
        return ModuleNode(
            type="module",
            path=x.path,
            args=[_subst_vars_in_any(a, env) for a in (x.args or [])],
            kwargs={k: _subst_vars_in_any(v, env) for k, v in (x.kwargs or {}).items()},
        )

    if isinstance(x, list):
        return [_subst_vars_in_any(v, env) for v in x]
    if isinstance(x, dict):
        return {k: _subst_vars_in_any(v, env) for k, v in x.items()}

    return x


def _expand_template_cfg(cfg: Config) -> Config:
    """
    Expand template-based tests into concrete tests.

    Template tests are expanded into executable test cases with resolved variables
    and stable ids, while non-template tests are passed through unchanged.
    """
    new_tests: List[TestItem] = []

    def _merge_common_with_overrides(
        *, common: Dict[str, Any], side_args: Optional[List[Any]], side_kwargs: Optional[Dict[str, Any]]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Merge common args/kwargs with side-specific overrides for compare-pair templates."""
        c_args = common.get("args", [])
        c_kwargs = common.get("kwargs", {})

        if c_args is None:
            c_args = []
        if c_kwargs is None:
            c_kwargs = {}

        if not isinstance(c_args, list):
            raise ValueError("template_compare_pair.common.args must be a list")
        if not isinstance(c_kwargs, dict):
            raise ValueError("template_compare_pair.common.kwargs must be a dict")

        args_out = list(c_args)
        kwargs_out = dict(c_kwargs)

        if side_args is not None:
            if not isinstance(side_args, list):
                raise ValueError("template_compare_pair side.args must be a list when provided")
            args_out = list(side_args)

        if side_kwargs is not None:
            if not isinstance(side_kwargs, dict):
                raise ValueError("template_compare_pair side.kwargs must be a dict when provided")
            kwargs_out.update(side_kwargs)

        args_out = [parse_node(x) for x in args_out]
        kwargs_out = {k: parse_node(v) for k, v in kwargs_out.items()}
        return args_out, kwargs_out

    # template_module -> expand to ModuleNode testcases
    for t in cfg.tests:
        if isinstance(t.op, TemplateModuleNode):
            op = t.op
            keys_all = sorted(op.vars.keys())
            base_id = t.id or "template"
            cases = op.cases if op.cases else [{}]

            for case in cases:
                # env0 = dict(case)
                env0 = dict(cfg.dims)
                env0.update(case)
                remaining_keys = [k for k in keys_all if k not in env0]
                remaining_lists = [op.vars[k] for k in remaining_keys]

                for combo in itertools.product(*remaining_lists) if remaining_keys else [()]:
                    env = dict(env0)
                    env.update(dict(zip(remaining_keys, combo)))

                    op_as_module = ModuleNode(type="module", path=op.path, args=op.args, kwargs=op.kwargs)
                    op2 = _subst_vars_in_any(op_as_module, env)

                    in2 = [_subst_vars_in_any(n, env) for n in t.in_]
                    kw2 = {k: _subst_vars_in_any(v, env) for k, v in (t.kwargs or {}).items()}

                    suffix = _id_suffix_format(keys_all, env)
                    t2 = t.model_copy(update={"id": f"{base_id}{suffix}", "op": op2, "in_": in2, "kwargs": kw2})
                    new_tests.append(t2)

            continue

       
        # template_compare_pair -> expand to TWO ModuleNode testcases (a then b) and wrap in "pair" testcase.
        if isinstance(t.op, TemplateComparePairNode):
            op = t.op
            keys_all = sorted(op.vars.keys())
            base_id = t.id or "template_compare_pair"
            cases = op.cases if op.cases else [{}]

            for case in cases:
                env0 = dict(cfg.dims)
                env0.update(case)
                remaining_keys = [k for k in keys_all if k not in env0]
                remaining_lists = [op.vars[k] for k in remaining_keys]

                for combo in itertools.product(*remaining_lists) if remaining_keys else [()]:
                    env = dict(env0)
                    env.update(dict(zip(remaining_keys, combo)))

                    a_args, a_kwargs = _merge_common_with_overrides(
                        common=op.common, side_args=op.a.args, side_kwargs=op.a.kwargs
                    )
                    b_args, b_kwargs = _merge_common_with_overrides(
                        common=op.common, side_args=op.b.args, side_kwargs=op.b.kwargs
                    )

                    a_mod = ModuleNode(type="module", path=op.a.path, args=a_args, kwargs=a_kwargs)
                    b_mod = ModuleNode(type="module", path=op.b.path, args=b_args, kwargs=b_kwargs)

                    a_mod2 = _subst_vars_in_any(a_mod, env)
                    b_mod2 = _subst_vars_in_any(b_mod, env)

                    in2 = [_subst_vars_in_any(n, env) for n in t.in_]
                    kw2 = {k: _subst_vars_in_any(v, env) for k, v in (t.kwargs or {}).items()}

                    suffix = _id_suffix_format(keys_all, env)

                    pair_id = f"{base_id}{suffix}"
                    tA = t.model_copy(
                        update={
                            "id": f"{pair_id}__a",
                            "impl": op.a.impl,
                            "op": a_mod2,
                            "in_": in2,
                            "kwargs": kw2,
                        }
                    )
                    tB = t.model_copy(
                        update={
                            "id": f"{pair_id}__b",
                            "impl": op.b.impl,
                            "op": b_mod2,
                            "in_": in2,
                            "kwargs": kw2,
                        }
                    )

                    new_tests.append(PairTest(id=pair_id, a=tA, b=tB))

            continue

        # non-template tests are copied as-is
        new_tests.append(t)

    return cfg.model_copy(update={"tests": new_tests})


def load_config_from_obj(raw_dict: Dict[str, Any]) -> Config:
    raw = _normalize_refs(raw_dict)

    presets = raw.get("presets", {}) or {}
    preset_keys = set(presets.keys()) if isinstance(presets, dict) else set()
    cfg = Config.model_validate(raw, context={"preset_keys": preset_keys})
    return _expand_template_cfg(cfg)


def load_config(path: str) -> Config:
    """
    Load and fully resolve a test configuration from a YAML file.

    This function performs the complete configuration loading pipeline:
    - Reads the YAML file from disk and recursively resolves any `include` directives.
    - Normalizes shorthand YAML syntax (e.g., ref/var shortcuts) into explicit typed forms.
    - Validates the resulting configuration against the Config schema.
    - Expands any template-based test definitions into concrete, executable test cases.

    Preset names are collected and provided as validation context to ensure
    references are checked during schema validation.

    Args:
        path: Filesystem path to the root YAML configuration file.

    Returns:
        A Config object with all includes resolved, templates expanded, and
        tests ready for execution.
    """
    raw = _load_raw_with_includes(path, _stack=set())
    raw = _normalize_refs(raw)

    presets = raw.get("presets", {}) or {}
    preset_keys = set(presets.keys()) if isinstance(presets, dict) else set()

    cfg = Config.model_validate(raw, context={"preset_keys": preset_keys})
    return _expand_template_cfg(cfg)

