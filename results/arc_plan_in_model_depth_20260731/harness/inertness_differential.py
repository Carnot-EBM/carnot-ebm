#!/usr/bin/env python3
"""Is the `plan_in_model` depth-label change INERT for every artifact that depends on it?

The freshness lint's acknowledgement mechanism demands that inertness be ESTABLISHED, not argued.
The claim is narrow: the change adds `depth_truncated_nodes` to `diagnostics` and can now write
`"depth_capped"` where `"queue_exhausted"` was written, but the RETURN VALUE -- the plan, or None
-- is identical on every path.

This proves it by DIFFERENTIAL EXECUTION rather than by reading the diff: the pre-change version
of the module is materialised from git HEAD into a separate module namespace, and both versions
are run over the same randomised corpus of (engine, goal, root, caps) with the returns compared
element-by-element. It also records, for each run, whether the termination label DIFFERED -- if it
never differed the corpus would not be exercising the change at all, and the "identical returns"
result would be vacuous.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

# Both module versions touch the induced-engine store at import time. Redirect it so neither
# version can write into `results/arc_e3` -- that tree is EVIDENCE (read, never write).
os.environ.setdefault("CARNOT_ARC_E3_DIR", tempfile.mkdtemp(prefix="e3_inertness_"))

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
REL = "python/carnot/agentic/arc_executable_world_model.py"
sys.path.insert(0, str(REPO / "python"))


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _corpus(rng, n):
    """Randomised worlds spanning the branches that matter: plan found, budget spent, depth
    capped, and genuinely exhausted. Engines are drawn from four mechanic families so the search
    tree shapes differ (chain, branch, no-op, and a shape-changing engine that exercises the
    `ng.shape != start.shape` reject)."""
    cases = []
    for i in range(n):
        n_cols = int(rng.integers(3, 14))
        family = int(rng.integers(0, 4))
        target = int(rng.integers(1, n_cols + 1))

        def make(family=family, target=target):
            def engine(grid, action, data):
                g = np.asarray(grid).copy()
                if family == 0:  # chain: fill leftmost zero
                    z = np.flatnonzero(g[0] == 0)
                    if z.size:
                        g[0, int(z[0])] = 1
                elif family == 1:  # branch: action-dependent write
                    idx = int(action) % g.shape[1]
                    g[0, idx] = (int(g[0, idx]) + 1 + int(action)) % 5 + 1
                elif family == 2:  # no-op
                    pass
                else:  # shape-changer on some actions
                    if int(action) % 3 == 0:
                        return g[:, :-1] if g.shape[1] > 1 else g
                    z = np.flatnonzero(g[0] == 0)
                    if z.size:
                        g[0, int(z[0])] = 1
                return g

            def is_done(grid):
                return bool(np.count_nonzero(np.asarray(grid)[0]) >= target)

            return engine, is_done

        # Named distinctly from the closures inside `make` -- reusing the names `engine` /
        # `is_done` here makes them loop variables that shadow the returned closures, which is a
        # real late-binding footgun in this exact shape even though `make`'s default-argument
        # capture already makes this instance safe.
        case_engine, case_is_done = make()
        cases.append(
            {
                "i": i,
                "engine": case_engine,
                "is_done": case_is_done,
                "root": np.zeros((1, n_cols), dtype=np.int16),
                "max_nodes": int(rng.choice([5, 25, 120, 2000, 20000])),
                "max_depth": int(rng.choice([1, 2, 4, 8, 40])),
                "goal_energy": (
                    (lambda g: -float(np.count_nonzero(np.asarray(g)[0])))
                    if bool(rng.integers(0, 2))
                    else None
                ),
            }
        )
    return cases


def _run(mod, c):
    diag: dict = {}
    try:
        plan = mod.plan_in_model(
            c["engine"],
            c["is_done"],
            c["root"],
            max_nodes=c["max_nodes"],
            max_depth=c["max_depth"],
            goal_energy=c["goal_energy"],
            diagnostics=diag,
        )
        # Normalise the return into something comparable across module instances.
        ret = None if plan is None else json.dumps(plan, sort_keys=True, default=str)
        return ret, diag.get("termination_reason"), None
    except Exception as exc:
        return f"RAISED {type(exc).__name__}: {exc}", diag.get("termination_reason"), True


def main(out_path: str) -> int:
    disk = REPO / REL
    now_sha = hashlib.sha256(disk.read_bytes()).hexdigest()
    head_bytes = subprocess.run(
        ["git", "show", f"HEAD:{REL}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    was_sha = hashlib.sha256(head_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as td:
        # Mirror the real package depth: the module does `Path(__file__).resolve().parents[3]`
        # at import time, so a flat temp file raises IndexError before any of this can run.
        head_dir = pathlib.Path(td) / "python" / "carnot" / "agentic"
        head_dir.mkdir(parents=True)
        head_path = head_dir / "arc_executable_world_model_head.py"
        head_path.write_bytes(head_bytes)
        old = _load(head_path, "_e3_head")
        new = _load(disk, "_e3_disk")

        rng = np.random.default_rng(20260731)
        cases = _corpus(rng, 600)
        t0 = time.time()
        n_ret_diff = n_label_diff = 0
        label_pairs: dict[str, int] = {}
        examples = []
        for c in cases:
            r_old, l_old, _ = _run(old, c)
            r_new, l_new, _ = _run(new, c)
            if r_old != r_new:
                n_ret_diff += 1
                if len(examples) < 5:
                    examples.append({"i": c["i"], "old": str(r_old)[:200], "new": str(r_new)[:200]})
            if l_old != l_new:
                n_label_diff += 1
            label_pairs[f"{l_old}->{l_new}"] = label_pairs.get(f"{l_old}->{l_new}", 0) + 1
        wall = round(time.time() - t0, 2)

    out = {
        "probe": "plan_in_model_depth_label_inertness_differential",
        "schema": "carnot.arc_plan_in_model_inertness.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "random_seed": 20260731,
        "reproducibility_checksum": hashlib.sha256(
            (was_sha + now_sha + "600_cases_seed_20260731").encode()
        ).hexdigest(),
        "path": REL,
        "sha256_was": was_sha,
        "sha256_now": now_sha,
        "n_cases": len(cases),
        "n_return_value_differences": n_ret_diff,
        "n_termination_label_differences": n_label_diff,
        "label_transitions": label_pairs,
        "return_examples_if_any": examples,
        "duration_s": wall,
        # The vacuity guard: if the corpus never triggered a relabel, "returns identical" would be
        # true of a corpus that never touched the change.
        "corpus_actually_exercises_the_change": n_label_diff > 0,
        "honest_verdict": (
            "complete_inert_returns_identical_across_600_differential_cases"
            if n_ret_diff == 0 and n_label_diff > 0
            else "complete_differential_found_a_return_value_difference_see_examples"
        ),
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": (
            "Correctness here is decided by executing both module versions and comparing returns "
            "directly -- the check IS the oracle. No verifier value-added or moat claim is made."
        ),
        "preconditions_checked": [
            {"resource": "git_head_version_of_module", "available": bool(head_bytes)},
        ],
    }
    pathlib.Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n_cases",
                    "n_return_value_differences",
                    "n_termination_label_differences",
                    "corpus_actually_exercises_the_change",
                    "label_transitions",
                )
            },
            indent=1,
        )
    )
    return 0 if n_ret_diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
