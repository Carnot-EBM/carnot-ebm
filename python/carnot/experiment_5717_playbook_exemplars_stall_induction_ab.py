"""Experiment 5717: matched-budget A/B of PLAYBOOK-EXEMPLAR injection on the STALL /
first-contact world-model induction (REQ-ARC-WMTE-5717).

WHAT IS BEING TESTED
--------------------
Phase 3 of the ARC exploration-playbook work adds a DEV-GATED capability
(``SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED`` / ``CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED``):
when the live agent's stall detector fires, it prepends a SMALL, game-AGNOSTIC few-shot of
exploration-method PATTERN statements (from
docs/research-notes/arc-exploration-playbook-20260717.md) to the tier-3 world-model
induction prompt, to bias the proposer's priors toward "simple general rules, don't
memorize coordinates, structural win condition, ..." WITHOUT asking it to reason at length
(exp5714 found long-reasoning induction overruns the token budget and emits zero code).

This is a GENUINELY OPEN empirical question, not a foregone conclusion. This experiment
measures whether the injection helps, hurts, or is inconclusive, on a matched budget.

DESIGN
------
For each roster game we solve L1 offline, replay the winning labels to collect the real
transition trajectory, and take a k-window ENDING at the L0->L1 boundary as induction
evidence (the same window contract exp5714 uses). Then, on the SAME window and the SAME
frozen live-stack proposer config (Qwen3.5-9B-MTP, codeonly ON, /no_think, matched
max_tokens/tries), we run two arms differing ONLY in the exemplar flag:

  - CONTROL   : proposer.include_playbook_exemplars = False  (byte-identical to production)
  - TREATMENT : proposer.include_playbook_exemplars = True   (the Phase-3 injection)

Each induced engine is scored by the executable WorldModelVerifier against the FULL
trajectory (a held-out-beyond-window generalization signal), plus induction_ok (did it emit
valid engine + is_level_complete) and induce wall time. We run TRIALS per (game, arm)
because the proposer samples at temperature>0 (non-deterministic), and aggregate.

HONESTY
-------
The proposer is stochastic; N is small (roster x trials). The verdict states the direction
AND flags the small sample explicitly. An inconclusive or negative result is the finding,
not a failure to hide. Substrate is a REAL local GGUF run (live_llm_inference).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

JsonDict = dict[str, Any]

RANDOM_SEED = 5717
# Games whose offline adapter reaches L1, so build_window() can capture a real L0->L1 window.
DEFAULT_ROSTER = ("lp85", "g50t", "cn04", "ft09")
WINDOW_K = 12
TRIALS_PER_ARM = 2
CUDA_GPU_INDEX = "1"  # outer-loop's RTX 3090 (24GB), per CLAUDE.md GPU allocation

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy tier-3 world-model induction proposer (frozen live stack)",
        "quant": "Q4_K_M",
    }
]

INFERENCE_SUBSTRATE = "live_llm_inference"

REQUIRED_RESULT_FIELDS = [
    "experiment",
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed self-declared state; lets the reconciler classify "
        "without re-running (Verdict Terminal-Prefix Discipline)."
    },
    "inference_substrate": {
        "principle": "explicit live_llm_inference declaration -> adversarial_verify applies the "
        "60s generative-inference duration floor; this loads + runs the real GGUF per arm."
    },
    "preconditions_checked": {
        "principle": "records WHICH resources were verified before any inference; pre-empts the "
        "silent-missing-resource fabrication mode (Pre-Launch Preconditions Discipline)."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility of the harness."},
    "reproducibility_checksum": {
        "principle": "content hash over config + per-arm results catches silent drift on replay."
    },
    "playbook_exemplars_delta_accuracy": {
        "principle": "treatment-minus-control mean reproduction accuracy; the primary "
        "induction-quality signal, honestly reported with its small-sample caveat."
    },
}


# --------------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------------
def preconditions() -> JsonDict:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf, _resolve_llama_server

    gguf = _resolve_gguf("Qwen3.5-9B-MTP") or _resolve_gguf("Qwen3.5-9B")
    server = _resolve_llama_server()
    checks = [
        {"resource": "qwen3.5_9b_mtp_gguf_cached", "available": bool(gguf)},
        {"resource": "llama_server_binary", "available": bool(server and Path(server).exists())},
    ]
    return {"gguf_path": gguf, "server_path": str(server), "preconditions_checked": checks}


def _first_precondition_miss(preconds: JsonDict) -> Optional[str]:
    for check in preconds["preconditions_checked"]:
        if not check["available"]:
            return str(check["resource"])
    return None


# --------------------------------------------------------------------------------------
# window + full-trajectory collection (adapts exp5714.build_levelup_window)
# --------------------------------------------------------------------------------------
def build_window(game: str, k: int = WINDOW_K) -> Optional[tuple[list, list, int]]:
    """Solve `game` to L1 offline, replay the winning labels, and return
    (induction_window, full_trajectory, cell): the k-window ending at the L0->L1 boundary
    (induction evidence) and EVERY transition on the winning path (held-out-beyond-window
    verification set). Returns None if L1 is not reached / no level-up captured."""
    import arc_loop_solve as loop
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import Transition, detect_cell, to_logical
    from carnot.agentic.arc_solver_kit import frame_level

    try:
        from carnot.experiment_5714_think_mode_rescoped_ab import _select_levelup_window
    except Exception:
        _select_levelup_window = None  # type: ignore[assignment]

    res = loop.solve_adaptered(game, 1)
    labels = res.get("solution_labels") or []
    if not labels or int(res.get("reached_level", 0)) < 1:
        return None
    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    cell = detect_cell(grid_of(f))
    if ad.warmup_label is not None:
        f = ad.apply(env, ad.warmup_label, f)
    prev_g = to_logical(grid_of(f), cell)
    prev_lvl = frame_level(f)
    trans: list = []
    for lbl in labels:
        f = ad.apply(env, lbl, f)
        g1 = to_logical(grid_of(f), cell)
        lvl = frame_level(f)
        act = (
            json.loads(lbl)
            if isinstance(lbl, str) and lbl.strip().startswith("{")
            else {"action": lbl}
        )
        trans.append(
            Transition(prev_g, int(act.get("action", 0)), act.get("data"), g1, prev_lvl, lvl)
        )
        prev_g, prev_lvl = g1, lvl
    if _select_levelup_window is not None:
        window = _select_levelup_window(trans, k)
    else:  # pragma: no cover - defensive fallback if the sibling helper moves
        levelups = [i for i, t in enumerate(trans) if t.level_after > t.level_before]
        window = trans[max(0, levelups[-1] - (k - 1)) : levelups[-1] + 1] if levelups else None
    if not window:
        return None
    return window, trans, cell


# --------------------------------------------------------------------------------------
# one induction arm on a shared window
# --------------------------------------------------------------------------------------
def run_arm(prop, game: str, arm_exemplars: bool, window: list, full: list, cell: int) -> JsonDict:
    """Run the FROZEN codeonly induce() with exemplars on/off, then score the induced engine
    on the full trajectory. arm_exemplars sets ONLY proposer.include_playbook_exemplars."""
    from carnot.agentic.arc_executable_world_model import (
        WorldModelVerifier,
        load_engine,
    )

    prop.include_playbook_exemplars = bool(arm_exemplars)
    t0 = time.time()
    try:
        ok, detail = prop.induce(game, window, cell)
    except Exception as exc:
        return {
            "game": game,
            "exemplars": bool(arm_exemplars),
            "induction_ok": False,
            "error": repr(exc)[:200],
            "induce_s": round(time.time() - t0, 1),
        }
    induce_s = round(time.time() - t0, 1)
    row: JsonDict = {
        "game": game,
        "exemplars": bool(arm_exemplars),
        "induction_ok": bool(ok),
        "induce_s": induce_s,
        "reproduction_accuracy": None,
        "cell_recall": None,
    }
    if not ok:
        row["induction_failure_detail"] = str(detail)[:200]
        return row
    try:
        engine, _is_done = load_engine(game)
        vr = WorldModelVerifier(full).score(engine)
        row["reproduction_accuracy"] = round(float(vr.accuracy), 4)
        row["n_transitions_scored"] = int(vr.n)
        row["n_correct"] = int(vr.n_correct)
        cell_recall = getattr(vr, "cell_recall", None)
        if cell_recall is not None:
            row["cell_recall"] = round(float(cell_recall), 4)
    except Exception as exc:
        row["verify_error"] = repr(exc)[:200]
    return row


# --------------------------------------------------------------------------------------
# aggregation + verdict
# --------------------------------------------------------------------------------------
def _arm_summary(rows: list[JsonDict], exemplars: bool) -> JsonDict:
    arm = [r for r in rows if r.get("exemplars") is exemplars]
    ok = [r for r in arm if r.get("induction_ok")]
    accs = [r["reproduction_accuracy"] for r in ok if r.get("reproduction_accuracy") is not None]
    return {
        "runs": len(arm),
        "induction_ok": len(ok),
        "induction_ok_rate": round(len(ok) / len(arm), 4) if arm else 0.0,
        "scored_runs": len(accs),
        "mean_reproduction_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "mean_induce_s": round(sum(r["induce_s"] for r in arm) / len(arm), 1) if arm else None,
    }


def _verdict(control: JsonDict, treatment: JsonDict, n_runs: int) -> tuple[str, Optional[float]]:
    c = control["mean_reproduction_accuracy"]
    t = treatment["mean_reproduction_accuracy"]
    if c is None or t is None:
        return "complete_playbook_exemplars_ab_no_scored_runs_inconclusive", None
    delta = round(t - c, 4)
    # A small, honest margin: with a stochastic proposer and small N this is directional only.
    if delta > 0.02 and treatment["induction_ok_rate"] >= control["induction_ok_rate"] - 0.001:
        v = f"complete_playbook_exemplars_improved_induction_accuracy_delta_{delta}_small_n_{n_runs}"
    elif delta < -0.02:
        v = f"complete_playbook_exemplars_hurt_induction_accuracy_delta_{delta}_small_n_{n_runs}"
    else:
        v = f"complete_playbook_exemplars_inconclusive_delta_{delta}_small_n_{n_runs}"
    return v, delta


def _checksum(payload: JsonDict) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def run(roster: tuple[str, ...] = DEFAULT_ROSTER, trials: int = TRIALS_PER_ARM) -> JsonDict:
    started = time.time()
    preconds = preconditions()
    base: JsonDict = {
        "experiment": "exp5717-playbook-exemplars-stall-induction-ab",
        "req": "REQ-ARC-WMTE-5717",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "random_seed": RANDOM_SEED,
        "roster": list(roster),
        "trials_per_arm": trials,
        "window_k": WINDOW_K,
        "field_provenance": FIELD_PRINCIPLES,
        "gguf_path": preconds["gguf_path"],
        "preconditions_checked": preconds["preconditions_checked"],
    }
    miss = _first_precondition_miss(preconds)
    if miss:
        base["honest_verdict"] = f"complete: blocked_{miss}"
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum({"blocked": miss, "roster": list(roster)})
        return base

    # Pin the generator to the outer-loop's 3090 (guarded; yields if the conductor holds it).
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", CUDA_GPU_INDEX)

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # DEDICATED port (default 8931, distinct from the live agents' 8919/8921) so this A/B spawns
    # its OWN server on the guarded CUDA GPU (CARNOT_ARC_GENERATOR_CUDA_GPU) instead of contending
    # with other agents' shared iGPU servers. MTP defaults ON (frozen-stack faithful) but is
    # togglable OFF via CARNOT_ARC_AB_MTP=0 for llama.cpp builds without --spec-type draft-mtp;
    # MTP is lossless speculative decoding so induction QUALITY is unchanged either way.
    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=preconds["gguf_path"],
        mtp=(os.environ.get("CARNOT_ARC_AB_MTP", "1") != "0"),
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        port=int(os.environ.get("CARNOT_ARC_AB_PORT", "8931")),
        max_tokens=int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
        timeout=int(os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "600")),
    )
    base["server_port"] = prop.port
    base["mtp_enabled"] = bool(prop.mtp)

    windows: dict[str, tuple[list, list, int]] = {}
    skipped: list[str] = []
    for game in roster:
        try:
            got = build_window(game)
        except Exception as exc:
            got = None
            skipped.append(f"{game}:build_error:{repr(exc)[:80]}")
        if got is None:
            skipped.append(f"{game}:no_l1_window")
            continue
        windows[game] = got

    rows: list[JsonDict] = []
    if not windows:
        base["honest_verdict"] = "complete_playbook_exemplars_ab_no_solvable_windows_blocked"
        base["skipped"] = skipped
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum({"skipped": skipped})
        return base

    # Interleave arms per game/trial so slow drift affects both arms equally (matched).
    for game, (window, full, cell) in windows.items():
        for trial in range(trials):
            for exemplars in (False, True):
                row = run_arm(prop, game, exemplars, window, full, cell)
                row["trial"] = trial
                rows.append(row)

    control = _arm_summary(rows, exemplars=False)
    treatment = _arm_summary(rows, exemplars=True)
    verdict, delta = _verdict(control, treatment, len(rows))

    base.update(
        {
            "honest_verdict": verdict,
            "skipped": skipped,
            "n_runs": len(rows),
            "control_exemplars_off": control,
            "treatment_exemplars_on": treatment,
            "playbook_exemplars_delta_accuracy": delta,
            "rows": rows,
            "methodology_note": (
                "The Qwen3.5-9B proposer samples at temperature>0, so per-arm induction is "
                "non-deterministic; N = len(roster) x trials x 2 arms is small. The delta is "
                "DIRECTIONAL, not a significance claim. Both arms share the identical window, "
                "proposer config, and budget; only include_playbook_exemplars differs. Verified "
                "against the full winning trajectory (held-out beyond the k-window)."
            ),
            "verifier_is_oracle": False,
            "duration_s": round(time.time() - started, 3),
        }
    )
    base["reproducibility_checksum"] = _checksum(
        {"rows": rows, "roster": list(roster), "trials": trials}
    )
    return base


def main() -> None:
    roster = DEFAULT_ROSTER
    trials = TRIALS_PER_ARM
    if len(sys.argv) > 1:
        roster = tuple(sys.argv[1].split(","))
    if len(sys.argv) > 2:
        trials = int(sys.argv[2])
    result = run(roster, trials)
    out = REPO_ROOT / "results" / "experiment_5717_playbook_exemplars_stall_induction_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"verdict: {result.get('honest_verdict')}")
    print(f"control:   {result.get('control_exemplars_off')}")
    print(f"treatment: {result.get('treatment_exemplars_on')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
