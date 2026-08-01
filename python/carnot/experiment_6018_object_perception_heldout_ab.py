#!/usr/bin/env python3
"""LEVER #1 (REQ-ARC-WMTE-5830) OBJECT-PERCEPTION INDUCTION A/B -- the heldout-accuracy
measurement `ops/verifier_gaps.md` has recorded as PENDING since 2026-07-24.

WHAT IS BEING MEASURED. `objects_block()` appends a connected-component OBJECT table
(translation-invariant `object_hash`, containment, adjacency) to the world-model induction
prompt behind `CARNOT_ARC_OBJECT_PERCEPTION` (default OFF). The question is whether that
object-structured view of the SAME frames makes the induced executable world model predict
transitions it was NOT shown. Two arms differ ONLY by that env flag: same generator server,
same window, same budget, same code path (`LocalGGUFProposer.induce` -> `induce_prompt`).

WHY A NEW HARNESS AND NOT `experiment_5831_object_perception_induction_ab.py`. That harness
(never completed -- its GPU wedged, see the gap entry) grades with
`WorldModelVerifier(list(window))`, i.e. on the WHOLE window including the transitions the
prompt SHOWED the model. `_transitions_block` shows `changed[:k-2] + noop[:2]` = 6
transitions at the default k=8, so on any game whose window has <= 6 transitions the
"heldout_accuracy" it reports is pure TRAINING accuracy with nothing held out at all.
Measured on the real windows: 6 of the 20 buildable games (r11l, lp85, cd82, sp80, ft09,
vc33) have ZERO held-out transitions, and 4 of the 8 games in exp5831's own DEFAULT_ROSTER
are in that set or fail to build. This harness therefore computes the held-out set as the
PROMPT COMPLEMENT and reports the two strata separately, so a training number can never be
read as a generalization number.

HOW THE HELD-OUT SET IS IDENTIFIED (read, not modelled). `_transitions_block`'s selection
rule is inline and not exported. Rather than reimplement it -- this project has twice been
burned by two reconstructions of one wrong formula agreeing with each other -- each
transition's own line is RENDERED with the same `_rle_delta_compact` the prompt uses and
tested for membership in the ACTUAL prompt string returned by `induce_prompt`. A transition
is held out iff its line does not appear in the prompt the model was given.

UNMEASURABILITY IS NOT CLEANLINESS (pre-registered). Qwen3.5-9B is known to floor at
heldout 0.0 on this task (the one prior data point, gap entry 2026-07-24). If the primary
metric is exactly 0.0 in BOTH arms on every cell, this run reports
`unmeasurable_instrument_floor` -- NOT "no difference" -- because a metric that cannot vary
cannot detect an effect. Games whose held-out set is empty are reported as EXCLUDED with a
reason, never as 0.0.

STATISTICS. Pre-registered before the run (see `--pre-register`): primary = per-game mean
held-out exact-full-grid accuracy, paired across arms, two-sided exact sign test plus an
exact sign-flip (Wilcoxon-style) permutation test. The MINIMUM REACHABLE two-sided p at the
planned support is stated in the pre-registration file and re-derived in the artifact; a
zero-discordance outcome is reported as "no test was possible", never as "no significant
difference".

inference_substrate: live_llm_inference (a real local GGUF is loaded and generates).
verifier_is_oracle: False -- the grader is exact-match against recorded transitions; the win
oracle (the level counter) is never consulted here.
solve_provenance: development_proxy -- this is a measurement of an offline induction metric,
not a live-agent self-discovery solve. Nothing is submitted; the flag stays default-off
regardless of the outcome (only the operator graduates it).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402


# THE CAP THIS EXPERIMENT WAS MEASURED AGAINST (pinned 2026-08-01, behaviour-preserving).
# This module's held-out design is `full \ shown`: it pops CARNOT_ARC_INDUCE_TRANSITIONS_K to
# get "the production prompt", derives the withheld indices from it, then RAISES the env as a
# positive control that shows the model every transition. That only works while the production
# default withholds something. On 2026-08-01 the default became "show every transition", which
# would have made `held` empty -- and `if not held: continue` would then have skipped the
# positive control SILENTLY, turning a leak check into a no-op without failing.
# 8 was the production default when this experiment ran and is what its published artifact
# measured, so pinning it keeps the result reproducible rather than tracking a default that has
# since moved underneath it.
_PRODUCTION_K_AT_MEASUREMENT = 8

EXPERIMENT_ID = 6018
REQUIREMENT = "REQ-ARC-WMTE-5830"
RANDOM_SEED = 6018
OUT_DIR = ROOT / "results" / "arc_object_perception_ab_20260728"
ARTIFACT = ROOT / "results" / "experiment_6018_object_perception_heldout_ab.json"
PREREG = OUT_DIR / "preregistration.json"

# The 20 games whose offline level-up induction window builds (5 of the 25 public games do
# not build one -- recorded, not silently dropped). Split by whether the window has any
# transition the prompt does NOT show, which is the only stratum where a HELD-OUT number
# exists at all.
HELDOUT_GAMES = [
    "ls20",
    "s5i5",
    "tu93",
    "cn04",
    "m0r0",
    "sk48",
    "ar25",
    "tr87",
    "g50t",
    "re86",
    "bp35",
    "sb26",
    "lf52",
    "su15",
]
TRAIN_ONLY_GAMES = ["r11l", "lp85", "cd82", "sp80", "ft09", "vc33"]
UNBUILDABLE_GAMES = {
    "wa30": "build_progress_window raised AttributeError: 'NoneType' has no attribute 'hand_verifier'",
    "sc25": "build_progress_window raised AttributeError: 'NoneType' has no attribute 'hand_verifier'",
    "tn36": "build_progress_window raised AttributeError: 'NoneType' has no attribute 'hand_verifier'",
    "ka59": "build_progress_window raised ValueError: invalid literal for int(): 'C:1'",
    "dc22": "build_progress_window returned None (no offline L1 window)",
}
N_REPLICATES = int(os.environ.get("OPAB_REPLICATES", "6"))
WALL_BUDGET_S = float(os.environ.get("OPAB_WALL_BUDGET_S", "9000"))


# ---------------------------------------------------------------------------
# statistics (pure; unit-tested in tests/python/test_object_perception_ab_stats.py)
# ---------------------------------------------------------------------------


def min_reachable_two_sided_p(n_discordant: int) -> float:
    """The SMALLEST two-sided p an exact sign test can return with `n_discordant` non-tied
    pairs. 1.0 when nothing is discordant -- which is the honest statement "no test was
    possible", NOT "no significant difference". Stated before the run so a null cannot be
    dressed up afterwards as a negative result at some implied power."""
    if n_discordant <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**n_discordant))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _binom_tail_ge(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / float(2**n)


def sign_test_two_sided(deltas: list[float]) -> dict[str, Any]:
    """Exact two-sided sign test on paired deltas. Ties (delta == 0) are dropped, and the
    dropped count is REPORTED -- a run whose every pair ties has no test, and the caller
    must be able to see that rather than read a p of 1.0 as evidence of no effect."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    d = pos + neg
    if d == 0:
        return {
            "n_pairs": len(deltas),
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": ties,
            "n_discordant": 0,
            "p_two_sided": 1.0,
            "test_was_possible": False,
            "min_reachable_two_sided_p_at_this_discordance": 1.0,
        }
    p = min(1.0, 2.0 * _binom_tail_ge(max(pos, neg), d))
    return {
        "n_pairs": len(deltas),
        "n_positive": pos,
        "n_negative": neg,
        "n_ties": ties,
        "n_discordant": d,
        "p_two_sided": round(p, 8),
        "test_was_possible": True,
        "min_reachable_two_sided_p_at_this_discordance": round(min_reachable_two_sided_p(d), 8),
    }


def signflip_exact_two_sided(deltas: list[float], *, max_n: int = 20) -> dict[str, Any]:
    """Exact sign-flip permutation test on the paired deltas (the magnitude-aware companion
    to the sign test: it uses the observed deltas, not just their signs). Enumerates all
    2^n sign assignments when n <= max_n; returns test_was_possible False otherwise so a
    caller can never mistake a skipped test for a passed one."""
    nz = [float(d) for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return {
            "n_nonzero": 0,
            "p_two_sided": 1.0,
            "test_was_possible": False,
            "observed_mean": 0.0,
        }
    if n > max_n:
        return {
            "n_nonzero": n,
            "p_two_sided": None,
            "test_was_possible": False,
            "observed_mean": round(sum(nz) / n, 8),
            "note": f"exact enumeration skipped: n={n} > max_n={max_n}",
        }
    obs = abs(sum(nz))
    hits = 0
    for mask in range(1 << n):
        s = 0.0
        for i, v in enumerate(nz):
            s += v if (mask >> i) & 1 else -v
        if abs(s) >= obs - 1e-12:
            hits += 1
    return {
        "n_nonzero": n,
        "p_two_sided": round(hits / float(1 << n), 8),
        "test_was_possible": True,
        "observed_mean": round(sum(nz) / n, 8),
        "n_enumerated": 1 << n,
    }


def bootstrap_ci(
    values: list[float], *, seed: int, n_resamples: int = 10000, alpha: float = 0.05
) -> dict[str, Any]:
    """Percentile bootstrap interval for the mean of `values`, resampling the UNIT OF
    INDEPENDENCE the caller passes in (per-GAME means here, not per-cell rows: replicates
    of one game share a window and are not independent draws)."""
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return {
        "mean": round(sum(values) / n, 8),
        "lo": round(lo, 8),
        "hi": round(hi, 8),
        "n": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# held-out identification (reads the REAL prompt; does not reimplement the selection)
# ---------------------------------------------------------------------------


def transition_prompt_line(e3: Any, t: Any) -> str:
    """The exact line `_transitions_block` emits for transition `t`, rendered with the same
    `_rle_delta_compact`. Used for MEMBERSHIP testing against the real prompt."""
    click = f" data={t.data}" if t.data else ""
    return (
        f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
        f"changed cells (FULL, run-length) = {e3._rle_delta_compact(t.grid, t.next_grid)}"
    )


def split_shown_heldout(
    e3: Any, game: str, window: list, cell: int, prompt: str
) -> tuple[list[int], list[int]]:
    """(shown_indices, heldout_indices) for `window` against the ACTUAL `prompt`."""
    shown, held = [], []
    for i, t in enumerate(window):
        (shown if transition_prompt_line(e3, t) in prompt else held).append(i)
    return shown, held


# ---------------------------------------------------------------------------
# GPU residency + wedge detection
# ---------------------------------------------------------------------------


def _nvidia(query: str, extra: str = "csv,noheader,nounits") -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-{query}", f"--format={extra}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return ""


def gpu_uuid_to_index() -> dict[str, int]:
    out = {}
    for line in _nvidia("gpu=index,uuid").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                out[parts[1]] = int(parts[0])
            except ValueError:
                continue
    return out


def per_pid_vram() -> list[dict]:
    idx = gpu_uuid_to_index()
    rows = []
    for line in _nvidia("compute-apps=pid,used_memory,gpu_uuid").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                rows.append(
                    {"pid": int(parts[0]), "mib": int(parts[1]), "gpu_index": idx.get(parts[2], -1)}
                )
            except ValueError:
                continue
    return rows


class VramMonitor:
    """Samples PER-PID VRAM + /health while the run proceeds, so "the model fell off the
    card" is a RECORDED FACT with a timestamp instead of surfacing later as an unexplained
    hang. The 2026-07-24 attempt at this same A/B had no such channel: GPU 1 went 21GB ->
    4MiB mid-run and the only symptom was a hung HTTP request."""

    def __init__(self, pid: int, health_url: str, path: Path, interval_s: float = 10.0):
        self.pid, self.health_url, self.path, self.interval_s = pid, health_url, path, interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.collapse_events: list[dict] = []
        self.health_failures: list[dict] = []
        self.baseline_mib: Optional[int] = None

    def _health(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self.health_url, timeout=10) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _sample(self) -> dict:
        rows = [r for r in per_pid_vram() if r["pid"] == self.pid]
        mib = rows[0]["mib"] if rows else 0
        gpu = rows[0]["gpu_index"] if rows else -1
        ok = self._health()
        s = {
            "t": round(time.time(), 2),
            "pid_mib": mib,
            "gpu_index": gpu,
            "health_ok": ok,
            "pid_present": bool(rows),
        }
        if self.baseline_mib is None and mib > 1000:
            self.baseline_mib = mib
        if self.baseline_mib and mib < 0.5 * self.baseline_mib:
            self.collapse_events.append(dict(s, baseline_mib=self.baseline_mib))
        if not ok:
            self.health_failures.append(s)
        return s

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = self._sample()
                self.samples.append(s)
                with self.path.open("a") as fh:
                    fh.write(json.dumps(s) + "\n")
            except Exception as exc:  # noqa: BLE001
                self.samples.append(
                    {
                        "t": round(time.time(), 2),
                        "sampler_error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def summary(self) -> dict:
        mibs = [s["pid_mib"] for s in self.samples if "pid_mib" in s]
        return {
            "server_pid": self.pid,
            "n_samples": len(self.samples),
            "interval_s": self.interval_s,
            "gpu_indices_observed": sorted(
                {s.get("gpu_index") for s in self.samples if s.get("gpu_index") is not None}
            ),
            "pid_mib_min": min(mibs) if mibs else None,
            "pid_mib_max": max(mibs) if mibs else None,
            "pid_mib_last": mibs[-1] if mibs else None,
            "baseline_mib": self.baseline_mib,
            "n_collapse_events": len(self.collapse_events),
            "collapse_events": self.collapse_events[:8],
            "n_health_failures": len(self.health_failures),
            "health_failures": self.health_failures[:8],
            "samples_path": str(self.path),
        }


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------


def build_preregistration() -> dict:
    planned_cells = (len(HELDOUT_GAMES) + len(TRAIN_ONLY_GAMES)) * N_REPLICATES * 2
    return {
        "experiment": f"experiment_{EXPERIMENT_ID}_object_perception_heldout_ab",
        "requirement": REQUIREMENT,
        "written_before_any_llm_call": True,
        "arms": {
            "control": "CARNOT_ARC_OBJECT_PERCEPTION unset/0",
            "treatment": "CARNOT_ARC_OBJECT_PERCEPTION=1",
        },
        "generator": "Qwen3.5-9B-MTP GGUF (operator-directed: NOT the 31B, whose 21GB on a "
        "24GB card is the documented cause of the 2026-07-24 wedge)",
        "primary_metric": "held-out exact-full-grid accuracy (WorldModelVerifier.accuracy on "
        "the PROMPT-COMPLEMENT transitions), averaged per game over replicates",
        "primary_support_games": list(HELDOUT_GAMES),
        "n_primary_support_games": len(HELDOUT_GAMES),
        "n_replicates_per_game_per_arm": N_REPLICATES,
        "planned_induce_calls": planned_cells,
        "primary_test": "two-sided exact sign test on per-game paired deltas; ties dropped",
        "secondary_test": "exact sign-flip permutation test on the same per-game deltas",
        "min_reachable_two_sided_p_if_all_support_games_discordant": round(
            min_reachable_two_sided_p(len(HELDOUT_GAMES)), 10
        ),
        "min_reachable_two_sided_p_formula": "2 * 0.5**n_discordant, capped at 1.0; "
        "1.0 when n_discordant == 0 (no test possible)",
        "secondary_metrics": [
            "held-out cell_recall",
            "held-out change_fidelity",
            "held-out correct_changed_cells",
            "held-out spurious_changed_cells",
            "induce_ok rate",
            "production tail-split (_split_prefix_heldout) accuracy",
            "full-window accuracy (the exp5831 quantity, for continuity)",
            "prompt chars",
            "generated tokens",
            "wall seconds",
        ],
        "train_only_stratum_games": list(TRAIN_ONLY_GAMES),
        "train_only_stratum_reason": "window has <= 6 transitions so the prompt shows ALL of "
        "them; any accuracy on these games is TRAINING accuracy "
        "and is reported separately, never as heldout",
        "excluded_games": UNBUILDABLE_GAMES,
        "floor_rule": "if the primary metric is exactly 0.0 in BOTH arms on EVERY cell, the "
        "verdict is unmeasurable_instrument_floor -- an instrument that cannot "
        "vary has not measured a null",
        "prior_data_point": "one Qwen-9B tu93 induce, heldout 0.000 for BOTH arms "
        "(ops/verifier_gaps.md 2026-07-24; explicitly not an A/B)",
        "declared_outcome_policy": "an honest null is reported plainly as a null; the flag "
        "stays default-off either way (operator-only graduation)",
        "random_seed": RANDOM_SEED,
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _score_row(e3: Any, engine: Any, transitions: list) -> dict:
    if not transitions:
        return {"measurable": False, "n": 0}
    vr = e3.WorldModelVerifier(list(transitions)).score(engine)
    return {
        "measurable": True,
        "n": int(vr.n),
        "n_correct": int(vr.n_correct),
        "accuracy": round(float(vr.accuracy), 6),
        "cell_recall": round(float(vr.cell_recall), 6),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "error": vr.error,
    }


def run() -> int:  # noqa: C901 - one linear measurement procedure, kept in one place
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cells").mkdir(exist_ok=True)

    # ---- PRECONDITIONS (Pre-Launch Preconditions Discipline) ----------------
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    gguf = e3._resolve_gguf("Qwen3.5-9B-MTP")
    conductor = subprocess.run(
        ["systemctl", "--user", "is-active", "carnot-conductor.service"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    # PROBE the treatment on a real (if tiny) transition, not on an empty list: the empty
    # list raises IndexError inside `objects_block` (`trans[0].grid`) because the function's
    # documented "any failure returns ''" try/except wraps only the blob_topology IMPORT, not
    # the body. That is NOT reachable from the live path (`induce_prompt` dereferences
    # `trans[0]` before ever calling objects_block), so it is recorded as an observation, not
    # fixed here -- but probing with [] measured the probe, not the treatment.
    probe_grid = np.zeros((8, 8), dtype=int)
    probe_next = probe_grid.copy()
    probe_next[2, 3] = 5
    probe_trans = [
        e3.Transition(
            grid=probe_grid,
            action=1,
            data=None,
            next_grid=probe_next,
            level_before=0,
            level_after=0,
        )
    ]
    try:
        objects_probe = e3.objects_block(probe_trans, previous_level_complete_grid=None)
        objects_importable = isinstance(objects_probe, str) and "OBJECTS" in objects_probe
    except Exception:  # noqa: BLE001
        objects_importable = False
    gpu_index = int(os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU", "0"))
    free_mb = e3._cuda_gpu_free_mb(gpu_index)
    need_mb = e3._generator_cuda_min_free_mb()
    # A generator is available either because a healthy server is ALREADY listening on the
    # requested port (the warm-reuse case: its VRAM is on the card, so the card correctly
    # reports NO headroom for a SECOND one -- 24 GiB cannot hold two 13.5 GiB servers) or
    # because the card has room to launch one. Checking only headroom would refuse to run
    # against the very server it is going to reuse.
    want_port = int(os.environ.get("OPAB_PORT", "0"))
    warm = False
    if want_port:
        import urllib.request

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{want_port}/health", timeout=10) as r:
                warm = r.status == 200
        except Exception:  # noqa: BLE001
            warm = False
    pre = [
        {"resource": "qwen3.5-9b-mtp_gguf_cached", "available": bool(gguf), "detail": str(gguf)},
        {
            "resource": "conductor_inactive",
            "available": conductor != "active",
            "detail": f"systemctl is-active -> {conductor!r}",
        },
        {
            "resource": "generator_launchable_or_already_warm",
            "available": bool(warm or free_mb >= need_mb),
            "detail": f"warm_server_on_port_{want_port}={warm} "
            f"cuda_gpu_{gpu_index}_free={free_mb} need={need_mb}",
        },
        {"resource": "objects_block_importable", "available": objects_importable},
        {
            "resource": "object_perception_flag_default_off",
            "available": not e3._object_perception_on(),
        },
    ]
    if not all(p["available"] for p in pre):
        missing = [p["resource"] for p in pre if not p["available"]]
        art = {
            "experiment": f"experiment_{EXPERIMENT_ID}_object_perception_heldout_ab",
            "requirement": REQUIREMENT,
            "honest_verdict": "blocked_precondition_" + "_".join(missing)[:120],
            "inference_substrate": "live_llm_inference",
            "preconditions_checked": pre,
            "duration_s": round(time.time() - t_start, 3),
        }
        ARTIFACT.write_text(json.dumps(art, indent=2))
        print("BLOCKED:", missing)
        return 0

    # ---- pre-registration, written and hashed BEFORE the first LLM call -----
    prereg = build_preregistration()
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    if PREREG.exists() and PREREG.read_text() != prereg_text:
        print(
            "NOTE: pre-registration on disk differs from the current plan; keeping DISK "
            "version as the binding one and recording both hashes."
        )
    else:
        PREREG.write_text(prereg_text)
    prereg_on_disk = PREREG.read_text()
    prereg_sha = "sha256:" + hashlib.sha256(prereg_on_disk.encode()).hexdigest()
    print(f"pre-registration {PREREG} {prereg_sha}")
    print(
        "  min reachable two-sided p if all "
        f"{len(HELDOUT_GAMES)} support games are discordant: "
        f"{prereg['min_reachable_two_sided_p_if_all_support_games_discordant']}"
    )

    # ---- windows (built once; shared by every arm so the flag is the only variable) ----
    windows: dict[str, tuple] = {}
    window_meta: dict[str, dict] = {}
    for game in HELDOUT_GAMES + TRAIN_ONLY_GAMES:
        t0 = time.time()
        try:
            w = atp.build_progress_window(game)
        except Exception as exc:  # noqa: BLE001
            print(f"  window FAILED {game}: {type(exc).__name__}: {exc}")
            window_meta[game] = {"built": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        if w is None:
            window_meta[game] = {"built": False, "error": "no_window"}
            continue
        windows[game] = w
        win, _full, cell = w
        p_off = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
        shown, held = split_shown_heldout(e3, game, list(win), cell, p_off)
        prefix, tail = wmte._split_prefix_heldout(list(win))
        window_meta[game] = {
            "built": True,
            "build_s": round(time.time() - t0, 2),
            "cell": int(cell),
            "n_transitions": len(win),
            "shown_indices": shown,
            "heldout_indices": held,
            "n_heldout": len(held),
            "n_production_tail": len(tail),
            "n_changing": sum(1 for t in win if not np.array_equal(t.grid, t.next_grid)),
        }
        print(
            f"  window {game}: n={len(win)} shown={len(shown)} heldout={len(held)} "
            f"tail={len(tail)} ({window_meta[game]['build_s']}s)"
        )

    # ---- TREATMENT WITNESS: the ON prompt must really differ ----------------
    treatment_witness = []
    for game, (win, _full, cell) in windows.items():
        os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        assert not e3._object_perception_on()
        p_off = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
        os.environ["CARNOT_ARC_OBJECT_PERCEPTION"] = "1"
        assert e3._object_perception_on()
        p_on = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
        os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        treatment_witness.append(
            {
                "game": game,
                "prompt_chars_off": len(p_off),
                "prompt_chars_on": len(p_on),
                "object_header_in_off": "OBJECT STRUCTURE" in p_off,
                "object_header_in_on": "OBJECT STRUCTURE" in p_on,
                "off_is_prefix_of_on": p_on.startswith(p_off),
                "object_block_chars": len(p_on) - len(p_off),
            }
        )

    # ---- generator ---------------------------------------------------------
    port = want_port or e3._free_port()
    prop = e3.LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=port,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=4096,
        timeout=1200,
    )
    if not prop._ensure_server():
        art = {
            "experiment": f"experiment_{EXPERIMENT_ID}_object_perception_heldout_ab",
            "requirement": REQUIREMENT,
            "honest_verdict": "blocked_generator_server_would_not_start",
            "inference_substrate": "live_llm_inference",
            "preconditions_checked": pre,
            "duration_s": round(time.time() - t_start, 3),
        }
        ARTIFACT.write_text(json.dumps(art, indent=2))
        print("BLOCKED: server would not start")
        return 0
    server_pid = getattr(prop._proc, "pid", None)
    if server_pid is None:  # reused a server we did not launch: find its pid by port
        lsof = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False).stdout
        for line in lsof.splitlines():
            if f":{prop.port} " in line and "pid=" in line:
                server_pid = int(line.split("pid=")[1].split(",")[0])
                break
    time.sleep(4)
    residency = [r for r in per_pid_vram() if r["pid"] == server_pid]
    print(
        f"server pid={server_pid} port={prop.port} residency={residency} "
        f"observed_n_ctx={prop.observed_n_ctx()}"
    )

    mon = VramMonitor(
        int(server_pid or -1),
        f"http://127.0.0.1:{prop.port}/health",
        OUT_DIR / "vram_samples.jsonl",
    )
    mon.start()

    # ---- cells ------------------------------------------------------------
    rows: list[dict] = []
    not_run: list[dict] = []
    wedged = False
    order = [(g, "heldout") for g in HELDOUT_GAMES if g in windows] + [
        (g, "train_only") for g in TRAIN_ONLY_GAMES if g in windows
    ]
    for rep in range(N_REPLICATES):
        # ALTERNATE the within-pair arm order by replicate so a warm-cache / ordering
        # advantage cannot accumulate on one arm.
        arms = (("off", "0"), ("on", "1")) if rep % 2 == 0 else (("on", "1"), ("off", "0"))
        for game, stratum in order:
            win, _full, cell = windows[game]
            for arm, flag in arms:
                if time.time() - t_start > WALL_BUDGET_S:
                    not_run.append(
                        {"game": game, "rep": rep, "arm": arm, "reason": "wall_budget_exhausted"}
                    )
                    continue
                if not prop._healthy():
                    wedged = True
                    not_run.append(
                        {
                            "game": game,
                            "rep": rep,
                            "arm": arm,
                            "reason": "generator_unhealthy_before_cell",
                        }
                    )
                    continue
                cell_id = f"{game}__r{rep}__{arm}"
                e3_dir = OUT_DIR / "e3" / cell_id
                cell_path = OUT_DIR / "cells" / f"{cell_id}.json"
                if cell_path.exists():
                    # RESUME, not re-measure: a completed cell is evidence already on disk.
                    rows.append(json.loads(cell_path.read_text()))
                    continue
                os.environ["CARNOT_ARC_OBJECT_PERCEPTION"] = flag
                assert e3._object_perception_on() is (flag == "1"), "arm flag not in effect"
                e3.E3_DIR = e3_dir  # per-cell isolation: no cell can read another's engine
                prompt = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
                shown, held = split_shown_heldout(e3, game, list(win), cell, prompt)
                sf0, cf0 = prop.n_server_failures, prop.n_content_failures
                t0 = time.time()
                try:
                    ok, msg = prop.induce(game, list(win), cell)
                    err = None
                except Exception as exc:  # noqa: BLE001
                    ok, msg, err = False, "", f"{type(exc).__name__}: {exc}"[:200]
                wall = time.time() - t0
                row: dict[str, Any] = {
                    "cell_id": cell_id,
                    "game": game,
                    "stratum": stratum,
                    "replicate": rep,
                    "arm": arm,
                    "object_perception_flag": flag,
                    "arm_order_in_replicate": [a for a, _ in arms],
                    "e3_dir": str(e3_dir),
                    "elapsed_s": round(wall, 2),
                    "induce_ok": bool(ok),
                    "induce_msg": str(msg)[:200],
                    "exception": err,
                    "prompt_chars": len(prompt),
                    "object_header_in_prompt": "OBJECT STRUCTURE" in prompt,
                    "n_shown": len(shown),
                    "n_heldout": len(held),
                    "stop_type": prop.last_stop_type,
                    "prompt_truncated": bool(prop.last_prompt_truncated),
                    "generated_tokens": int(prop.last_generated_tokens),
                    "raw_completion_chars": len(prop.last_raw_completion),
                    "server_failures_delta": prop.n_server_failures - sf0,
                    "content_failures_delta": prop.n_content_failures - cf0,
                    "server_healthy_after": prop._healthy(),
                }
                engine = ilc = None
                if ok:
                    try:
                        engine, ilc = e3.load_engine(game)
                    except Exception as exc:  # noqa: BLE001
                        row["load_engine_error"] = f"{type(exc).__name__}: {exc}"[:200]
                row["engine_loaded"] = engine is not None
                row["is_level_complete_present"] = ilc is not None
                wm = e3_dir / game / "world_model.py"
                if wm.exists():
                    body = wm.read_bytes()
                    row["engine_sha256"] = hashlib.sha256(body).hexdigest()
                    row["engine_bytes"] = len(body)
                if engine is not None:
                    heldout_trans = [win[i] for i in held]
                    _prefix, tail = wmte._split_prefix_heldout(list(win))
                    row["heldout"] = _score_row(e3, engine, heldout_trans)
                    row["production_tail"] = _score_row(e3, engine, list(tail))
                    row["full_window"] = _score_row(e3, engine, list(win))
                else:
                    row["heldout"] = {"measurable": False, "n": len(held), "reason": "no_engine"}
                    row["production_tail"] = {"measurable": False, "reason": "no_engine"}
                    row["full_window"] = {"measurable": False, "reason": "no_engine"}
                rows.append(row)
                cell_path.write_text(json.dumps(row, indent=2))
                acc = (row["heldout"] or {}).get("accuracy")
                print(
                    f"[r{rep} {game:5s} {arm:3s}] ok={ok} heldout_n={len(held)} "
                    f"acc={acc} cr={(row['heldout'] or {}).get('cell_recall')} "
                    f"cf={(row['heldout'] or {}).get('change_fidelity')} {wall:.1f}s"
                )
            os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        if wedged:
            break

    mon.stop()
    os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
    (OUT_DIR / "rows.json").write_text(json.dumps(rows, indent=2))
    analysis = analyse(rows, window_meta)
    (OUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2))

    art = build_artifact(
        rows=rows,
        analysis=analysis,
        window_meta=window_meta,
        treatment_witness=treatment_witness,
        preconditions=pre,
        prereg=prereg,
        prereg_sha=prereg_sha,
        monitor=mon.summary(),
        residency=residency,
        server_pid=server_pid,
        port=prop.port,
        gguf=str(gguf),
        n_ctx=prop.n_ctx,
        observed_n_ctx=prop.observed_n_ctx(),
        not_run=not_run,
        wedged=wedged,
        liveness=prop.liveness_witness(),
        duration_s=time.time() - t_start,
    )
    ARTIFACT.write_text(json.dumps(art, indent=2, default=str))
    print("\n" + json.dumps(analysis["primary"], indent=2))
    print("verdict:", art["honest_verdict"])
    print("wrote", ARTIFACT)
    return 0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

_HELDOUT_FIELDS = (
    "accuracy",
    "cell_recall",
    "change_fidelity",
    "correct_changed_cells",
    "spurious_changed_cells",
    "n_changes_correct",
)


def _paired_per_game(
    rows: list[dict], stratum: str, block: str, field: str
) -> dict[str, dict[str, list[float]]]:
    """{game: {"off": [...], "on": [...], "replicates": [...]}} over MATCHED replicates only.

    MATCHED means: a replicate contributes only if BOTH arms measured `field` in it. This is
    not fussiness -- an any-cell union is how a 22-cell arm got compared against a 25-cell
    control in an earlier run of this project and produced a phantom difference. If a run is
    truncated (wall budget, a wedged generator) the unmatched tail is DROPPED from the paired
    comparison rather than silently averaged into one arm.
    """
    by_cell: dict[tuple[str, int, str], float] = {}
    games: set[str] = set()
    reps: dict[str, set[int]] = {}
    for r in rows:
        if r.get("stratum") != stratum:
            continue
        blk = r.get(block) or {}
        if not blk.get("measurable") or field not in blk:
            continue
        by_cell[(r["game"], int(r["replicate"]), r["arm"])] = float(blk[field])
        games.add(r["game"])
        reps.setdefault(r["game"], set()).add(int(r["replicate"]))
    out: dict[str, dict[str, list[float]]] = {}
    for game in sorted(games):
        offs, ons, kept = [], [], []
        for rep in sorted(reps.get(game, ())):
            o = by_cell.get((game, rep, "off"))
            n = by_cell.get((game, rep, "on"))
            if o is None or n is None:
                continue
            offs.append(o)
            ons.append(n)
            kept.append(rep)
        if kept:
            out[game] = {"off": offs, "on": ons, "replicates": [float(k) for k in kept]}
    return out


def analyse(rows: list[dict], window_meta: dict) -> dict:
    res: dict[str, Any] = {"n_rows": len(rows)}

    # induce success (measurable on every cell, both strata)
    for stratum in ("heldout", "train_only"):
        srows = [r for r in rows if r.get("stratum") == stratum]
        res[f"induce_ok_{stratum}"] = {
            arm: {
                "n_cells": sum(1 for r in srows if r["arm"] == arm),
                "n_ok": sum(1 for r in srows if r["arm"] == arm and r["induce_ok"]),
                "n_engine_loaded": sum(
                    1 for r in srows if r["arm"] == arm and r.get("engine_loaded")
                ),
            }
            for arm in ("off", "on")
        }

    per_field = {}
    for field in _HELDOUT_FIELDS:
        pg = _paired_per_game(rows, "heldout", "heldout", field)
        games, deltas, means = [], [], {}
        for game, arms in sorted(pg.items()):
            if not arms.get("off") or not arms.get("on"):
                continue
            m_off = sum(arms["off"]) / len(arms["off"])
            m_on = sum(arms["on"]) / len(arms["on"])
            games.append(game)
            deltas.append(m_on - m_off)
            means[game] = {
                "off": round(m_off, 6),
                "on": round(m_on, 6),
                "delta": round(m_on - m_off, 6),
                "n_matched_replicates": len(arms["off"]),
                "matched_replicates": [int(r) for r in arms["replicates"]],
            }
        all_cells = [
            v for arms in pg.values() for k, vals in arms.items() if k != "replicates" for v in vals
        ]
        floored = bool(all_cells) and all(v == 0.0 for v in all_cells)
        per_field[field] = {
            "per_game_means": means,
            "games": games,
            "mean_delta_over_games": round(sum(deltas) / len(deltas), 8) if deltas else None,
            "sign_test": sign_test_two_sided(deltas),
            "signflip_test": signflip_exact_two_sided(deltas),
            "bootstrap_ci_over_games": bootstrap_ci(deltas, seed=RANDOM_SEED),
            "all_cells_exactly_zero_both_arms": floored,
            "n_cells_contributing": len(all_cells),
            "n_distinct_cell_values": len(sorted({round(v, 8) for v in all_cells})),
        }
    res["heldout_by_field"] = per_field
    res["multiplicity_note"] = (
        f"{len(_HELDOUT_FIELDS)} held-out channels are tested. Only `accuracy` is the "
        "PRE-REGISTERED primary; the other five are exploratory secondaries, and a p below "
        f"0.05 on one of them should be read against a Bonferroni threshold of "
        f"{round(0.05 / len(_HELDOUT_FIELDS), 5)}, not against 0.05. They are also strongly "
        "correlated with each other (all are functions of the same predicted grids), so "
        "Bonferroni is conservative and the honest reading of any secondary hit is 'worth one "
        "confirmatory run', not 'established'."
    )

    # UNMATCHED-CELL LEDGER: a cell whose (game, replicate) partner in the other arm is
    # missing cannot enter a paired comparison. Counted and NAMED rather than averaged into
    # one arm (the phantom-difference failure mode).
    def _unmatched(stratum: str, block: str) -> list[str]:
        seen = {
            (r["game"], int(r["replicate"]), r["arm"])
            for r in rows
            if r.get("stratum") == stratum and (r.get(block) or {}).get("measurable")
        }
        return sorted(
            f"{g}__r{rep}__{arm}"
            for (g, rep, arm) in seen
            if (g, rep, "on" if arm == "off" else "off") not in seen
        )

    unmatched = _unmatched("heldout", "heldout")
    res["unmatched_cells_excluded_from_pairing"] = unmatched
    res["n_unmatched_cells"] = len(unmatched)
    # SCOPE, STATED. The gate keys on the HELD-OUT stratum because that is what the paired
    # primary claim rests on. The train-only stratum is counted separately rather than left
    # out of the ledger silently -- an unmatched cell there is a real (if benign) fact: the
    # generated module raised at import so no engine could be scored, and its replicate is
    # dropped from that stratum's pairing.
    unmatched_train = _unmatched("train_only", "full_window")
    res["unmatched_cells_train_only_stratum"] = unmatched_train
    res["n_unmatched_cells_train_only_stratum"] = len(unmatched_train)
    res["cells_with_no_loadable_engine"] = sorted(
        f"{r['cell_id']}: {str(r.get('load_engine_error'))[:80]}"
        for r in rows
        if r.get("induce_ok") and not r.get("engine_loaded")
    )
    res["primary"] = {
        "metric": "held-out exact-full-grid accuracy (prompt-complement transitions)",
        **{k: v for k, v in per_field["accuracy"].items() if k != "per_game_means"},
        "per_game_means": per_field["accuracy"]["per_game_means"],
    }

    # continuity blocks: the production tail split, and the exp5831 full-window quantity
    for block, label in (
        ("production_tail", "production_tail_split_accuracy"),
        ("full_window", "full_window_accuracy_exp5831_quantity"),
    ):
        pg = _paired_per_game(rows, "heldout", block, "accuracy")
        deltas, means = [], {}
        for game, arms in sorted(pg.items()):
            if not arms.get("off") or not arms.get("on"):
                continue
            m_off = sum(arms["off"]) / len(arms["off"])
            m_on = sum(arms["on"]) / len(arms["on"])
            deltas.append(m_on - m_off)
            means[game] = {
                "off": round(m_off, 6),
                "on": round(m_on, 6),
                "delta": round(m_on - m_off, 6),
            }
        res[label] = {
            "per_game_means": means,
            "mean_delta_over_games": round(sum(deltas) / len(deltas), 8) if deltas else None,
            "sign_test": sign_test_two_sided(deltas),
        }

    # train-only stratum (TRAINING accuracy -- never a generalization claim)
    pg = _paired_per_game(rows, "train_only", "full_window", "accuracy")
    deltas, means = [], {}
    for game, arms in sorted(pg.items()):
        if not arms.get("off") or not arms.get("on"):
            continue
        m_off = sum(arms["off"]) / len(arms["off"])
        m_on = sum(arms["on"]) / len(arms["on"])
        deltas.append(m_on - m_off)
        means[game] = {
            "off": round(m_off, 6),
            "on": round(m_on, 6),
            "delta": round(m_on - m_off, 6),
        }
    res["train_only_training_accuracy"] = {
        "note": "these games' prompts SHOW every transition in the window; this is TRAINING "
        "accuracy and is NOT evidence of generalization",
        "per_game_means": means,
        "mean_delta_over_games": round(sum(deltas) / len(deltas), 8) if deltas else None,
        "sign_test": sign_test_two_sided(deltas),
    }

    # cost side (always measurable)
    for name, key in (
        ("prompt_chars", "prompt_chars"),
        ("elapsed_s", "elapsed_s"),
        ("generated_tokens", "generated_tokens"),
    ):
        res[f"cost_{name}"] = {
            arm: round(
                sum(float(r[key]) for r in rows if r["arm"] == arm)
                / max(1, sum(1 for r in rows if r["arm"] == arm)),
                3,
            )
            for arm in ("off", "on")
        }

    # DEAD-CHANNEL CENSUS: every analysis field, how many distinct values it took.
    census = []
    for field in _HELDOUT_FIELDS:
        info = per_field[field]
        census.append(
            {
                "field": f"heldout.{field}",
                "n_cells": info["n_cells_contributing"],
                "n_distinct_values": info["n_distinct_cell_values"],
                "constant": info["n_distinct_cell_values"] <= 1,
                "all_zero": info["all_cells_exactly_zero_both_arms"],
            }
        )
    res["field_population_census"] = census
    res["floored_fields"] = [c["field"] for c in census if c["constant"]]
    res["n_channels_that_varied"] = sum(1 for c in census if not c["constant"])
    res["why_the_census_is_a_finding_and_NOT_an_acceptance_gate"] = (
        "`floored_fields` is DERIVED from `field_population_census` in this same function, so "
        "a gate asserting 'every constant field is declared floored' could never fail on real "
        "data -- it would compare a list against a list computed from its own input. That is "
        "the forced-gate shape this project shipped once already (a recount gated against a "
        "count derived from the same markers). The census is therefore REPORTED, and the gate "
        "that exists instead asks a question the data can actually answer NO to: did any "
        "channel vary at all (`n_channels_that_varied` > 0)? A run in which every recorded "
        "channel is constant has measured nothing, and must fail loudly rather than be "
        "reported as a null."
    )
    return res


def build_artifact(
    *,
    rows,
    analysis,
    window_meta,
    treatment_witness,
    preconditions,
    prereg,
    prereg_sha,
    monitor,
    residency,
    server_pid,
    port,
    gguf,
    n_ctx,
    observed_n_ctx,
    not_run,
    wedged,
    liveness,
    duration_s,
    reverification=None,
    positive_control=None,
) -> dict:
    primary = analysis["primary"]
    sign = primary["sign_test"]
    floored = primary["all_cells_exactly_zero_both_arms"]
    n_support = len(primary["games"])

    verdict = _verdict(floored, sign, primary, n_support, wedged)

    # ---- gates, each with a witness computed at its OWN level ---------------
    gates: list[dict[str, Any]] = []

    tw_ok = bool(treatment_witness) and all(
        w["object_header_in_on"]
        and not w["object_header_in_off"]
        and w["object_block_chars"] > 0
        and w["off_is_prefix_of_on"]
        for w in treatment_witness
    )
    gates.append(
        {
            "gate": "treatment_is_real_in_every_prompt",
            "principle": "an A/B whose arms produce the same prompt measures the control twice; "
            "exp6013 did exactly that with the HUD mask and reported it as two arms",
            "passed": tw_ok,
            "witness": {
                "n_games": len(treatment_witness),
                "n_with_object_header_on": sum(
                    1 for w in treatment_witness if w["object_header_in_on"]
                ),
                "n_with_object_header_off": sum(
                    1 for w in treatment_witness if w["object_header_in_off"]
                ),
                "object_block_chars_min": min(
                    (w["object_block_chars"] for w in treatment_witness), default=None
                ),
                "object_block_chars_max": max(
                    (w["object_block_chars"] for w in treatment_witness), default=None
                ),
            },
        }
    )

    dirs = [r["e3_dir"] for r in rows]
    gates.append(
        {
            "gate": "every_cell_wrote_its_own_isolated_engine_store",
            "principle": "arms sharing results/arc_e3/<game>/world_model.py overwrite each "
            "other's engines; a 2026-07-27 run had to be discarded for this",
            "passed": len(set(dirs)) == len(dirs) and len(dirs) == len(rows),
            "witness": {
                "n_rows": len(rows),
                "n_distinct_e3_dirs": len(set(dirs)),
                "n_distinct_engine_sha256": len(
                    {r.get("engine_sha256") for r in rows if r.get("engine_sha256")}
                ),
            },
        }
    )

    prereg_min_p = prereg["min_reachable_two_sided_p_if_all_support_games_discordant"]
    recomputed = round(min_reachable_two_sided_p(prereg["n_primary_support_games"]), 10)
    gates.append(
        {
            "gate": "min_reachable_p_was_stated_before_the_run",
            "principle": "a 0-discordance outcome must never be reported as 'no significant "
            "difference'; the reachable floor has to be fixed in advance",
            "passed": bool(prereg_sha) and prereg_min_p == recomputed,
            "witness": {
                "preregistration_sha256": prereg_sha,
                "preregistered_min_reachable_p": prereg_min_p,
                "recomputed_min_reachable_p": recomputed,
                "observed_min_reachable_p_at_this_discordance": sign[
                    "min_reachable_two_sided_p_at_this_discordance"
                ],
                "observed_n_discordant": sign["n_discordant"],
            },
        }
    )

    collapse_free = monitor["n_collapse_events"] == 0
    resident = bool(residency) and residency[0]["mib"] > 1000
    gates.append(
        {
            "gate": "generator_stayed_resident_on_the_card_it_was_pinned_to",
            "principle": "the 2026-07-24 attempt at this same A/B died when a model fell off the "
            "PCI bus; per-PID VRAM sampling makes that a recorded fact, not a hang",
            "passed": bool(
                resident and collapse_free and monitor["n_health_failures"] == 0 and not wedged
            ),
            "witness": {
                "server_pid": server_pid,
                "residency_at_start": residency,
                "pid_mib_min": monitor["pid_mib_min"],
                "pid_mib_max": monitor["pid_mib_max"],
                "pid_mib_last": monitor["pid_mib_last"],
                "gpu_indices_observed": monitor["gpu_indices_observed"],
                "n_vram_samples": monitor["n_samples"],
                "n_collapse_events": monitor["n_collapse_events"],
                "n_health_failures": monitor["n_health_failures"],
                "wedged": wedged,
            },
        }
    )

    census = analysis["field_population_census"]
    constant_fields = [c["field"] for c in census if c["constant"]]
    gates.append(
        {
            "gate": "at_least_one_recorded_channel_actually_varied",
            "principle": "a field that never varies is either a measured floor or a dead "
            "channel (a census found 877 stat blocks with an errors key and ZERO "
            "non-zero values). Asserting 'every constant field is declared floored' "
            "would be FORCED -- the declared list is derived from the census itself -- "
            "so the gate instead asks the one question this data can answer NO to: if "
            "EVERY channel is constant, the run measured nothing and must fail loudly "
            "instead of reporting a null.",
            "passed": analysis["n_channels_that_varied"] > 0,
            "witness": {
                "census": census,
                "n_channels_that_varied": analysis["n_channels_that_varied"],
                "n_channels_total": len(census),
                "constant_fields_found": constant_fields,
                "declared_floored_fields": analysis["floored_fields"],
            },
        }
    )

    # The floor gate must NOT compare the verdict against the same flag the verdict was
    # derived from -- that is a forced gate (found by mutation: flipping
    # analysis.primary.all_cells_exactly_zero_both_arms moved BOTH sides and the gate never
    # failed). The independent side is the RAW ROWS: recount the floor straight off the
    # recorded per-cell held-out accuracies, with no analysis in the path.
    raw_heldout_accs = [
        float((r.get("heldout") or {}).get("accuracy"))
        for r in rows
        if r.get("stratum") == "heldout"
        and (r.get("heldout") or {}).get("measurable")
        and _is_number((r.get("heldout") or {}).get("accuracy"))
    ]
    floored_from_rows = bool(raw_heldout_accs) and all(a == 0.0 for a in raw_heldout_accs)
    verdict_has_floor_token = "unmeasurable_instrument_floor" in verdict
    gates.append(
        {
            "gate": "floor_is_reported_as_unmeasurable_not_as_no_difference",
            "principle": "an instrument that returns the same value on every cell has not "
            "measured a null; the prior single Qwen-9B data point was 0.000 on both "
            "arms and this run must not launder that into a negative result. The "
            "floor is RECOUNTED from the raw cell rows so this gate cannot be "
            "satisfied by the analysis agreeing with itself.",
            "passed": floored_from_rows == verdict_has_floor_token,
            "witness": {
                "floor_recounted_from_raw_rows": floored_from_rows,
                "n_raw_heldout_accuracy_values": len(raw_heldout_accs),
                "n_raw_nonzero": sum(1 for a in raw_heldout_accs if a != 0.0),
                "analysis_reported_floor": floored,
                "analysis_agrees_with_raw_recount": floored == floored_from_rows,
                "verdict_contains_floor_token": verdict_has_floor_token,
                "n_cells_contributing": primary["n_cells_contributing"],
                "n_distinct_cell_values": primary["n_distinct_cell_values"],
            },
        }
    )

    pc = positive_control or {}
    gates.append(
        {
            "gate": "the_null_is_backed_by_a_positive_control_that_moves_the_metric",
            "principle": "FALSE_NEGATIVE_RISK: a null claim is not a finding unless a positive "
            "control shows the metric CAN move. Here the control shows the model the "
            "withheld transitions outright (k raised past the window size) and grades "
            "on exactly those. If accuracy stays 0.0 even then, this metric is "
            "unreachable for this model class and the main comparison says nothing "
            "about perception -- which is a materially different conclusion from "
            "'the object block does not help'.",
            "passed": bool(pc.get("metric_demonstrably_moves")),
            "witness": {
                "ran": bool(pc.get("ran")),
                "n_cells": pc.get("n_cells"),
                "max_accuracy": pc.get("max_accuracy"),
                "mean_accuracy": pc.get("mean_accuracy"),
                "n_cells_with_nonzero_accuracy": pc.get("n_cells_with_nonzero_accuracy"),
                "max_cell_recall": pc.get("max_cell_recall"),
                "n_cells_with_nonzero_cell_recall": pc.get("n_cells_with_nonzero_cell_recall"),
            }
            if pc
            else {"not_performed": True},
        }
    )

    gates.append(
        {
            "gate": "no_cell_entered_a_comparison_without_its_matched_partner",
            "principle": "comparing a 22-cell arm against a 25-cell control produced a phantom "
            "win difference in an earlier run of this project. Per-game means are "
            "taken over replicates where BOTH arms measured; any cell without a "
            "partner is dropped and NAMED here, and a run that had to drop cells "
            "(wall budget, wedged generator) fails this gate rather than reporting "
            "a clean paired result.",
            "passed": analysis["n_unmatched_cells"] == 0,
            "witness": {
                "scope": "the HELD-OUT stratum, which is what the paired primary claim rests "
                "on; the train-only stratum is counted separately below rather than "
                "omitted from the ledger",
                "n_unmatched_cells": analysis["n_unmatched_cells"],
                "unmatched_cells": analysis["unmatched_cells_excluded_from_pairing"][:24],
                "n_unmatched_cells_train_only_stratum": analysis[
                    "n_unmatched_cells_train_only_stratum"
                ],
                "unmatched_cells_train_only_stratum": analysis[
                    "unmatched_cells_train_only_stratum"
                ][:24],
                "cells_with_no_loadable_engine": analysis["cells_with_no_loadable_engine"][:24],
                "per_game_matched_replicates": {
                    g: m.get("n_matched_replicates")
                    for g, m in analysis["primary"]["per_game_means"].items()
                },
            },
        }
    )

    gates.append(
        {
            "gate": "every_recorded_number_re_derives_from_the_engine_on_disk",
            "principle": "the metric code being right does not make the BOOKKEEPING right; a "
            "row can record a number produced by a different engine or a different "
            "transition slice than it claims. Re-running the SAME production verifier "
            "against the SAME engine file on disk catches that class, which no amount "
            "of re-implementing the metric would.",
            "passed": bool(
                reverification
                and reverification.get("n_cells_rechecked", 0) > 0
                and reverification.get("n_mismatched", 1) == 0
            ),
            "witness": reverification
            or {"not_performed": True, "why": "no reverification block was supplied"},
        }
    )

    vacuous = [g["gate"] for g in gates if not g["witness"]]
    all_pass = all(g["passed"] for g in gates)

    art: dict[str, Any] = {
        "experiment": f"experiment_{EXPERIMENT_ID}_object_perception_heldout_ab",
        "experiment_id": EXPERIMENT_ID,
        "requirement": REQUIREMENT,
        "run_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "title": "LEVER #1 object-perception induction A/B on held-out transitions: the "
        "measurement REQ-ARC-WMTE-5830 has had PENDING since 2026-07-24.",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": {
            "value": False,
            "principle": "the grader is exact-match / changed-cell overlap against RECORDED "
            "transitions, not the executable win oracle; the level counter is "
            "never consulted, so a win here cannot be circular",
        },
        "solve_provenance": {
            "value": "development_proxy",
            "principle": "an offline induction-quality measurement on the dev twin; it is "
            "NOT evidence that the live agent self-discovered anything, and no "
            "level is claimed",
        },
        "random_seed": RANDOM_SEED,
        "measurement_wall_s": round(sum(float(r["elapsed_s"]) for r in rows), 3),
        "duration_s": round(duration_s, 3),
        "model_specs": [
            {
                "name": "Qwen3.5-9B-MTP-GGUF",
                "gguf_path": gguf,
                "role": "induction_generator",
                "server_pid": server_pid,
                "port": port,
                "declared_n_ctx": n_ctx,
                "observed_n_ctx": observed_n_ctx,
                "mtp": True,
                "kv_quant": "q8_0",
                "why_9b_not_31b": "operator directive 2026-07-28: the 31B (21GB on a 24GB eGPU "
                "3090) is the documented cause of the 2026-07-24 wedge that "
                "killed the prior attempt at this measurement; a model-class "
                "change is operator-only",
            }
        ],
        "preconditions_checked": preconditions,
        "preregistration": {"path": str(PREREG), "sha256": prereg_sha, "content": prereg},
        "treatment_witness_per_game": treatment_witness,
        "window_metadata": window_meta,
        "excluded_games": UNBUILDABLE_GAMES,
        "generator_liveness_witness": liveness,
        "vram_monitor": monitor,
        "gpu_residency_at_start": residency,
        "cells_not_run": not_run,
        "run_was_wedged": wedged,
        "n_cells": len(rows),
        "analysis": analysis,
        "reverification": reverification,
        "positive_control": positive_control,
        "findings": _findings(rows, analysis, window_meta, treatment_witness),
        # NAME: not `acceptance_gates`. scripts/summarize_artifact.py prints the full repr of
        # every top-level key containing "acceptance_gate", and this list carries the whole
        # per-gate witness (including the field census), which would bury the summary it
        # exists to provide. The short boolean/name keys below are the ones summarize reads.
        "verification_gates": gates,
        "acceptance_gate_failed_names": [g["gate"] for g in gates if not g["passed"]],
        "acceptance_gate_passed": all_pass,
        "acceptance_gate_vacuous": vacuous,
        "acceptance_gate_passed_and_none_vacuous": bool(all_pass and not vacuous),
        "REQUIRED_PHRASING": (
            "The primary comparison is reported with the number of DISCORDANT per-game pairs "
            "and the minimum reachable two-sided p at that discordance. If discordance is 0 "
            "the correct statement is 'no test was possible', NOT 'no significant "
            "difference'. If every cell returned exactly 0.0 in both arms the correct "
            "statement is 'the instrument floored and could not discriminate', NOT 'the "
            "object block does not help'."
        ),
        "methodology_note": (
            "Two arms toggle ONLY CARNOT_ARC_OBJECT_PERCEPTION around the real "
            "LocalGGUFProposer.induce -> induce_prompt path, against one warm Qwen3.5-9B-MTP "
            "server pinned to a single RTX 3090 (per-PID VRAM sampled throughout). Each cell "
            "writes its engine into its own CARNOT_ARC_E3_DIR so no cell can read another "
            "arm's engine. The held-out set is the PROMPT COMPLEMENT, identified by rendering "
            "each transition's own line with the same _rle_delta_compact the prompt uses and "
            "testing membership in the actual prompt string -- not by reimplementing "
            "_transitions_block's selection rule. Games whose window has <= 6 transitions "
            "show the model everything and are reported in a separate TRAINING-accuracy "
            "stratum. Within-pair arm order alternates by replicate. The sampler seed is NOT "
            "controlled: the shipped generate() path posts no 'seed' field to llama.cpp "
            "(arc_executable_world_model.py generate(), _payload), so replicates are "
            "independent draws at temperature 0.2+0.1*attempt and the pairing is by "
            "(game, replicate), not by a shared RNG stream. Replication, not seed matching, "
            "is what bounds that variance here."
        ),
        "not_submitted": {
            "value": True,
            "principle": "ARC/Kaggle submission is operator-only and the quota gate (an "
            "offline result beating both a TRM baseline and the best prior "
            "submitted run) is not met; nothing here is submitted and no "
            "SUBMITTED_* flag is touched",
        },
        "flag_remains_default_off": {
            "value": True,
            "principle": "CARNOT_ARC_OBJECT_PERCEPTION is a measurement lever, not a "
            "graduation; only the operator flips a default",
        },
    }
    art["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    )
    return art


def reverify_rows(rows: list[dict], windows: dict[str, tuple], *, e3: Any) -> dict:
    """PROVENANCE RE-DERIVATION: for every cell that loaded an engine, re-load THAT cell's
    engine from ITS OWN directory on disk, re-score it on that game's held-out transitions,
    and compare against the number the row recorded.

    This is not a second implementation of the metric (two reconstructions agreeing is not
    evidence -- this project has been burned by exactly that). It is the SAME production
    verifier re-run against the SAME artefact on disk, which catches the bookkeeping class of
    error the metric code cannot: a row whose numbers came from a different engine or a
    different transition slice than the row claims.
    """
    checked = mismatched = missing = 0
    details: list[dict] = []
    for r in rows:
        if not r.get("engine_loaded"):
            continue
        game = r["game"]
        if game not in windows:
            missing += 1
            continue
        win, _full, _cell = windows[game]
        prompt_held = window_heldout_indices(e3, game, windows[game])
        wm = Path(r["e3_dir"]) / game / "world_model.py"
        if not wm.exists():
            missing += 1
            continue
        sha = hashlib.sha256(wm.read_bytes()).hexdigest()
        try:
            engine, _ilc = e3._load_engine_from(Path(r["e3_dir"]), game)
        except Exception as exc:  # noqa: BLE001
            details.append({"cell_id": r["cell_id"], "load_error": f"{type(exc).__name__}"})
            mismatched += 1
            continue
        again = _score_row(e3, engine, [win[i] for i in prompt_held])
        checked += 1
        recorded = r.get("heldout") or {}
        diffs = {
            k: [recorded.get(k), again.get(k)]
            for k in ("n", "accuracy", "cell_recall", "change_fidelity", "correct_changed_cells")
            if recorded.get(k) != again.get(k)
        }
        if diffs or sha != r.get("engine_sha256"):
            mismatched += 1
            details.append(
                {
                    "cell_id": r["cell_id"],
                    "field_diffs": diffs,
                    "sha_matches": sha == r.get("engine_sha256"),
                }
            )
    return {
        "n_cells_rechecked": checked,
        "n_mismatched": mismatched,
        "n_skipped_no_engine_or_window": missing,
        "mismatch_details": details[:12],
        "what_this_checks": "the row's recorded held-out numbers re-derive from the engine "
        "file still on disk in that cell's own isolated directory, at the "
        "sha256 the row recorded",
    }


def window_heldout_indices(e3: Any, game: str, window_tuple: tuple) -> list[int]:
    win, _full, cell = window_tuple
    prompt = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
    _shown, held = split_shown_heldout(e3, game, list(win), cell, prompt)
    return held


def _findings(
    rows: list[dict], analysis: dict, window_meta: dict, treatment_witness: list[dict]
) -> list[dict]:
    """Findings this run produced that were NOT the question it was asked. Each one is
    computed from the recorded data, so a reader can re-derive it rather than take it on
    trust."""
    built = {g: m for g, m in window_meta.items() if m.get("built")}
    zero_heldout = sorted(g for g, m in built.items() if m.get("n_heldout", 0) == 0)
    exp5831_roster = ["ls20", "tu93", "r11l", "lp85", "sc25", "cd82", "sk48", "sp80"]
    exp5831_unmeasurable = sorted(
        g
        for g in exp5831_roster
        if g in UNBUILDABLE_GAMES or (g in built and built[g].get("n_heldout", 0) == 0)
    )
    off = [r for r in rows if r["arm"] == "off"]
    on = [r for r in rows if r["arm"] == "on"]

    def _mean(xs: list[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 3) if xs else None

    out = [
        {
            "finding": "the metric this project calls 'induction heldout_accuracy' is NOT "
            "held out on every game it was measured on",
            "evidence": {
                "games_with_zero_heldout_transitions": zero_heldout,
                "n_games_built": len(built),
                "exp5831_default_roster": exp5831_roster,
                "exp5831_roster_games_that_could_not_contribute_a_heldout_number": exp5831_unmeasurable,
            },
            "why_it_matters": "exp5831 grades with WorldModelVerifier(list(window)) -- the "
            "WHOLE window, including the transitions the prompt showed. On a "
            "window with <= 6 transitions the prompt shows all of them, so "
            "the reported 'heldout_accuracy' is TRAINING accuracy. Half of "
            "exp5831's own roster is in that state (or does not build at "
            "all), which is why this run reports the two strata separately.",
        },
        {
            "finding": "5 of the 25 public games cannot build an offline induction window at "
            "all, so the induction corpus covers 20 games, not 25",
            "evidence": {"excluded": UNBUILDABLE_GAMES},
            "why_it_matters": "two distinct pre-existing exceptions (a None adapter reaching "
            ".hand_verifier, and an int() parse of 'C:1'). Recorded, not "
            "fixed here -- fixing them would change the corpus mid-measurement.",
        },
        {
            "finding": "the object block is not free: it lengthens the prompt and the generation",
            "evidence": {
                "mean_prompt_chars_off": _mean([float(r["prompt_chars"]) for r in off]),
                "mean_prompt_chars_on": _mean([float(r["prompt_chars"]) for r in on]),
                "mean_elapsed_s_off": _mean([float(r["elapsed_s"]) for r in off]),
                "mean_elapsed_s_on": _mean([float(r["elapsed_s"]) for r in on]),
                "mean_generated_tokens_off": _mean([float(r["generated_tokens"]) for r in off]),
                "mean_generated_tokens_on": _mean([float(r["generated_tokens"]) for r in on]),
            },
            "why_it_matters": "a live agent under an action/latency budget pays this on every "
            "re-induction, so a null on quality is a NET NEGATIVE at equal "
            "quality, not a neutral result.",
        },
        {
            "finding": "objects_block's documented 'any failure returns \"\"' defence covers "
            "only the blob_topology import, not the body",
            "evidence": {
                "repro": "objects_block([]) raises IndexError at trans[0].grid",
                "reachable_from_live_path": False,
                "why_not_reachable": "induce_prompt dereferences trans[0].grid.shape "
                "before it ever calls objects_block, so an empty "
                "transition list fails upstream first",
            },
            "why_it_matters": "found while writing this harness's PRECONDITIONS probe, which "
            "initially probed with [] and therefore measured the probe rather "
            "than the treatment (the blocked artifact from that first attempt "
            "is preserved at "
            "results/arc_object_perception_ab_20260728/"
            "blocked_run_precondition_probe_bug.json). Not fixed here: it is "
            "a docstring-scope inaccuracy on an unreachable path, and this "
            "run must not change the code it is measuring.",
        },
    ]
    return out


def _verdict(floored: bool, sign: dict, primary: dict, n_support: int, wedged: bool) -> str:
    parts = ["complete_object_perception_heldout_ab"]
    if wedged:
        parts.append("PARTIAL_generator_wedged")
    if floored:
        parts.append("unmeasurable_instrument_floor_primary_zero_both_arms")
    else:
        md = primary.get("mean_delta_over_games")
        parts.append(f"mean_per_game_delta_{md}")
    if not sign["test_was_possible"]:
        parts.append("no_test_possible_zero_discordant_pairs")
    else:
        parts.append(f"sign_test_p_{sign['p_two_sided']}_on_{sign['n_discordant']}_discordant")
    parts.append(f"n_support_games_{n_support}")
    return "_".join(parts)


POSITIVE_CONTROL_PATH = OUT_DIR / "positive_control.json"


def run_positive_control(reps: int = 2) -> dict:
    """LEAK CONTROL: can this metric move AT ALL with this model?

    A null on "object perception does not lift held-out accuracy" is only informative if the
    held-out accuracy is a quantity this generator can move. So: induce the SAME games with
    `CARNOT_ARC_INDUCE_TRANSITIONS_K` raised so the prompt shows EVERY transition in the
    window -- including the ones the production k=8 prompt withholds -- and then grade on
    exactly the production-withheld indices. The model is being shown the answers.

    If accuracy is still 0.0 here, the metric is unreachable for this model class regardless of
    how the frames are serialized, and the main comparison is an UNINFORMATIVE null about
    perception (the binding constraint is elsewhere -- writing an engine that reproduces a
    64x64 transition exactly). If accuracy rises above 0.0, the metric demonstrably moves and
    the main null is about the object block, not about the instrument.

    The object-perception flag stays OFF throughout: this control is about the metric's
    reachability, not about the treatment.
    """
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3

    os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
    assert not e3._object_perception_on()
    port = int(os.environ.get("OPAB_PORT", "0")) or e3._free_port()
    prop = e3.LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=port,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=4096,
        timeout=1200,
    )
    if not prop._ensure_server():
        return {"ran": False, "reason": "generator_server_would_not_start"}
    rows: list[dict] = []
    t0 = time.time()
    for game in HELDOUT_GAMES:
        w = atp.build_progress_window(game)
        if w is None:
            continue
        win, _full, cell = w
        # the PRODUCTION-withheld indices, computed from the production k=8 prompt
        os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
        prod_prompt = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
        _shown, held = split_shown_heldout(e3, game, list(win), cell, prod_prompt)
        if not held:
            continue
        os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = str(len(win) + 4)
        leak_prompt = e3.induce_prompt(game, list(win), cell, k=_PRODUCTION_K_AT_MEASUREMENT)
        leak_shown, leak_held = split_shown_heldout(e3, game, list(win), cell, leak_prompt)
        for rep in range(reps):
            e3.E3_DIR = OUT_DIR / "e3_positive_control" / f"{game}__r{rep}"
            t = time.time()
            ok, msg = prop.induce(game, list(win), cell)
            row: dict[str, Any] = {
                "game": game,
                "replicate": rep,
                "induce_ok": bool(ok),
                "induce_msg": str(msg)[:160],
                "elapsed_s": round(time.time() - t, 2),
                "n_production_heldout": len(held),
                "n_shown_in_leak_prompt": len(leak_shown),
                "n_still_withheld_in_leak_prompt": len(leak_held),
                "leak_prompt_chars": len(leak_prompt),
                "graded_on": "the production-withheld indices, which THIS prompt showed",
            }
            if ok:
                try:
                    engine, _ilc = e3.load_engine(game)
                    row["scored"] = _score_row(e3, engine, [win[i] for i in held])
                except Exception as exc:  # noqa: BLE001
                    row["scored"] = {
                        "measurable": False,
                        "error": f"{type(exc).__name__}: {exc}"[:160],
                    }
            else:
                row["scored"] = {"measurable": False, "reason": "induce_failed"}
            rows.append(row)
            print(
                f"[PC {game} r{rep}] ok={ok} acc={(row['scored'] or {}).get('accuracy')} "
                f"cr={(row['scored'] or {}).get('cell_recall')} {row['elapsed_s']}s"
            )
        os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
    accs = [
        float(r["scored"]["accuracy"])
        for r in rows
        if (r.get("scored") or {}).get("measurable") and "accuracy" in r["scored"]
    ]
    recalls = [
        float(r["scored"]["cell_recall"])
        for r in rows
        if (r.get("scored") or {}).get("measurable") and "cell_recall" in r["scored"]
    ]
    out = {
        "ran": True,
        "what_it_is": "every transition shown in the prompt (k raised past the window size), "
        "graded on the indices the production k=8 prompt withholds",
        "n_cells": len(rows),
        "duration_s": round(time.time() - t0, 2),
        "max_accuracy": max(accs) if accs else None,
        "mean_accuracy": round(sum(accs) / len(accs), 6) if accs else None,
        "n_cells_with_nonzero_accuracy": sum(1 for a in accs if a > 0.0),
        "max_cell_recall": max(recalls) if recalls else None,
        "n_cells_with_nonzero_cell_recall": sum(1 for c in recalls if c > 0.0),
        "metric_demonstrably_moves": bool(accs and max(accs) > 0.0),
        "rows": rows,
    }
    POSITIVE_CONTROL_PATH.write_text(json.dumps(out, indent=2))
    return out


def main(argv: list[str]) -> int:
    if "--positive-control" in argv:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = run_positive_control()
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
        return 0
    if "--pre-register" in argv:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        prereg = build_preregistration()
        text = json.dumps(prereg, indent=2, sort_keys=True)
        PREREG.write_text(text)
        print(text)
        print("\nsha256:" + hashlib.sha256(text.encode()).hexdigest())
        return 0
    return run()


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main(sys.argv[1:]))
