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
from collections.abc import Callable, Sequence
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
        "principle": "treatment-minus-control mean EXACT-match reproduction accuracy; floors at 0 "
        "for single-shot first-contact induction, so it is reported but not the discriminator."
    },
    "playbook_exemplars_delta_cell_recall": {
        "principle": "treatment-minus-control mean GRADED per-changed-cell recall; the "
        "discriminating (non-flooring) induction-quality signal, directional under small N."
    },
    "metric_floored": {
        "principle": "bare bool true when even graded cell_recall is at its floor for both arms, "
        "so the offline metric cannot detect an effect -- an honest 'unmeasurable offline', NOT "
        "evidence the feature helps or hurts (mirrors the AUTO_HUD_MASK levels_gained floor)."
    },
    "outlier_fragile_direction": {
        "principle": "bare bool true when |delta| < the largest single-run cell_recall, so one "
        "lucky stochastic induction could account for or flip the direction; forbids calling "
        "improved/hurt on noise-dominated data under a temperature>0 proposer with small N."
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
def build_window_from_labels(
    game: str,
    labels: Sequence[str],
    apply: Callable[[Any, str, Any], Any],
    *,
    k: int = WINDOW_K,
    warmup_label: Optional[str] = None,
    label_to_action_data: Optional[Callable[[Any, str], tuple]] = None,
) -> Optional[tuple[list, list, int]]:
    """Replay `labels` offline and cut the induction window. NO GameAdapter required.

    WHY THIS EXISTS (2026-07-31). `build_window` took a game name and obtained its labels by
    calling `solve_adaptered`, which requires a registered GameAdapter. That coupling forced
    an adapter to exist for any game that was to appear in an induction corpus -- and for 18
    of the 25 public games the adapter's `action_labels` just replays a BANKED plan one label
    at a time. The verifier-routed search, hazard pruning, learned-verifier warm start and
    state dedup that `solve_adaptered` provides are all unexercised when the action set is a
    single forced label: the search is a straight line through a solution that was already
    known.

    So those 18 adapters were not solving anything. They were satisfying a function
    signature. `arc_solver_kit.reproduce()` already proves the coupling is unnecessary -- it
    takes `(labels, apply)` directly, and was used on 2026-07-31 to gate-verify wa30's entire
    670-action L9 route with no adapter at all.

    This function is that same contract for window building. A game with a banked route can
    now produce an induction window from the route directly, and `GameAdapter` can go back to
    meaning ONE thing: "this game has a searchable action space".

    Args:
        game: game id, used to open the offline env and for error messages.
        labels: the winning action labels, in order, from a fresh reset.
        apply: `(env, label, frame) -> frame`. `arc_game_adapters._default_json_apply`
            handles the common `{"action": N}` JSON label.
        k: window size; the returned window is the k transitions ending at the L0->L1 boundary.
        warmup_label: applied once before the route if the game consumes its first step
            (sc25 does).
        label_to_action_data: `(env, label) -> (action_id, data)` for games whose labels
            carry a payload (ka59's "C:<sprite_index>" click). Without it a non-integer
            label raises rather than silently recording action 0.

    Returns:
        (induction_window, full_trajectory, cell), or None if no level-up was captured.

    Spec: REQ-ARC-WMTE-5717
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import Transition, detect_cell, to_logical
    from carnot.agentic.arc_solver_kit import frame_level

    try:
        from carnot.experiment_5714_think_mode_rescoped_ab import _select_levelup_window
    except Exception:
        _select_levelup_window = None  # type: ignore[assignment]

    if not labels:
        return None

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    cell = detect_cell(grid_of(f))
    if warmup_label is not None:
        f = apply(env, warmup_label, f)
    prev_g = to_logical(grid_of(f), cell)
    prev_lvl = frame_level(f)

    trans: list = []
    for lbl in labels:
        f = apply(env, lbl, f)
        g1 = to_logical(grid_of(f), cell)
        lvl = frame_level(f)
        act = (
            json.loads(lbl)
            if isinstance(lbl, str) and lbl.strip().startswith("{")
            else {"action": lbl}
        )
        # Resolve the label to (action_id, data). `int(...)` alone is right for the games
        # whose labels ARE integers, but ka59 emits "C:<sprite_index>" for a click and this
        # line raised `ValueError: invalid literal for int(): 'C:1'` -- so ka59 produced NO
        # induction window and was silently missing from every corpus built here.
        #
        # Deliberately FAIL LOUD on an unparseable label rather than defaulting to action 0.
        # A Transition is the induction evidence the LLM reads; recording the wrong action
        # for a click would not crash, it would quietly teach the model a false dynamics,
        # which is far worse than a missing game.
        raw_action = act.get("action", 0)
        if label_to_action_data is not None and isinstance(raw_action, str):
            act_id, act_data = label_to_action_data(env, raw_action)
        else:
            try:
                act_id, act_data = int(raw_action), act.get("data")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{game}: cannot parse action label {raw_action!r} into (action, data). "
                    "Pass a `label_to_action_data` callable -- see ka59's, which maps "
                    "'C:<i>' to (6, {x,y})."
                ) from exc
        trans.append(Transition(prev_g, act_id, act_data, g1, prev_lvl, lvl))
        prev_g, prev_lvl = g1, lvl

    if _select_levelup_window is not None:
        window = _select_levelup_window(trans, k)
    else:  # pragma: no cover - defensive fallback if the sibling helper moves
        levelups = [i for i, t in enumerate(trans) if t.level_after > t.level_before]
        window = trans[max(0, levelups[-1] - (k - 1)) : levelups[-1] + 1] if levelups else None
    if not window:
        return None
    return window, trans, cell


def build_window(game: str, k: int = WINDOW_K) -> Optional[tuple[list, list, int]]:
    """Solve `game` to L1 offline via its GameAdapter, then cut the induction window.

    Thin caller over `build_window_from_labels` since 2026-07-31: this path is now only
    responsible for OBTAINING the labels (by search, through the adapter). The replay and
    window-cutting are shared, so a game with a banked route can skip the adapter entirely.
    See `build_window_from_labels` for why that separation matters.
    """
    import arc_loop_solve as loop
    from carnot.agentic import arc_game_adapters as adapters

    res = loop.solve_adaptered(game, 1)
    labels = res.get("solution_labels") or []
    if not labels or int(res.get("reached_level", 0)) < 1:
        return None
    ad = adapters.get_adapter(game)
    return build_window_from_labels(
        game,
        labels,
        ad.apply,
        k=k,
        warmup_label=ad.warmup_label,
        label_to_action_data=getattr(ad, "label_to_action_data", None),
    )


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
        # Graded per-changed-cell recall: a NON-flooring quality signal (an engine that gets some
        # changed cells right scores >0), unlike the strict exact-grid-match accuracy.
        row["cell_recall"] = round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4)
    except Exception as exc:
        row["verify_error"] = repr(exc)[:200]
    return row


# --------------------------------------------------------------------------------------
# aggregation + verdict
# --------------------------------------------------------------------------------------
# Below this graded-recall level, the offline induction-quality metric is effectively at its
# floor (a single-shot first-contact engine reproduces ~nothing of a hard game's trajectory),
# so it cannot discriminate the arms regardless of any real prompt effect.
FLOOR_CELL_RECALL = 0.05


def _arm_summary(rows: list[JsonDict], exemplars: bool) -> JsonDict:
    arm = [r for r in rows if r.get("exemplars") is exemplars]
    ok = [r for r in arm if r.get("induction_ok")]
    accs = [r["reproduction_accuracy"] for r in ok if r.get("reproduction_accuracy") is not None]
    recalls = [r["cell_recall"] for r in ok if r.get("cell_recall") is not None]
    return {
        "runs": len(arm),
        "induction_ok": len(ok),
        "induction_ok_rate": round(len(ok) / len(arm), 4) if arm else 0.0,
        "scored_runs": len(accs),
        "mean_reproduction_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "mean_cell_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "max_cell_recall": round(max(recalls), 4) if recalls else None,
        "mean_induce_s": round(sum(r["induce_s"] for r in arm) / len(arm), 1) if arm else None,
    }


def _arm_recalls(rows: list[JsonDict], exemplars: bool) -> list[float]:
    return [
        float(r["cell_recall"])
        for r in rows
        if r.get("exemplars") is exemplars
        and r.get("induction_ok")
        and r.get("cell_recall") is not None
    ]


def _leave_one_out_fragile(
    c_recalls: list[float], t_recalls: list[float], delta: float, threshold: float
) -> bool:
    """Leave-one-out robustness: is the direction of `delta` fragile to removing ONE run? Removes
    the single largest cell_recall (the run most able to inflate its arm's mean), recomputes the
    delta, and calls it fragile if the sign flips OR the magnitude drops to/below `threshold`. This
    correctly separates a real, tight separation (removing any one run barely moves the mean -> NOT
    fragile) from a one-lucky-run artifact (removing that run flips the sign -> fragile). Arms with
    fewer than 2 scored runs cannot establish robustness, so they are treated as fragile."""
    if len(c_recalls) < 2 or len(t_recalls) < 2:
        return True
    c_max, t_max = max(c_recalls), max(t_recalls)
    if c_max >= t_max:
        c2 = list(c_recalls)
        c2.remove(c_max)
        d2 = sum(t_recalls) / len(t_recalls) - sum(c2) / len(c2)
    else:
        t2 = list(t_recalls)
        t2.remove(t_max)
        d2 = sum(t2) / len(t2) - sum(c_recalls) / len(c_recalls)
    sign_flipped = (delta >= 0) != (d2 >= 0)
    return sign_flipped or abs(d2) <= threshold


def _verdict(
    control: JsonDict, treatment: JsonDict, rows: list[JsonDict], n_runs: int
) -> tuple[str, Optional[float], bool, bool]:
    """Return (verdict, cell_recall_delta, metric_floored, outlier_fragile). Exact-grid-match
    accuracy floors at 0 for single-shot first-contact induction, so we discriminate on the graded
    cell_recall AND (1) report when even that is at its floor, and (2) refuse to call a direction
    that a single high-variance run could flip (leave-one-out) -- the stochastic proposer produces
    per-run cell_recall from ~0 to ~0.7, so an outlier-driven mean is not a reliable direction."""
    cr_c = control["mean_cell_recall"]
    cr_t = treatment["mean_cell_recall"]
    if cr_c is None or cr_t is None:
        return "complete_playbook_exemplars_ab_no_scored_runs_inconclusive", None, True, False
    delta = round(cr_t - cr_c, 4)
    floored = max(cr_c, cr_t) < FLOOR_CELL_RECALL
    outlier_fragile = _leave_one_out_fragile(
        _arm_recalls(rows, exemplars=False), _arm_recalls(rows, exemplars=True), delta, 0.02
    )
    if floored:
        # Both arms at the metric floor -> the offline induction-quality metric cannot detect any
        # effect (mirrors the AUTO_HUD_MASK levels_gained floor). NOT evidence the feature is bad.
        v = f"complete_playbook_exemplars_metric_floored_inconclusive_cellrecall_delta_{delta}_n_{n_runs}"
    elif outlier_fragile:
        v = f"complete_playbook_exemplars_no_reliable_signal_high_variance_cellrecall_delta_{delta}_n_{n_runs}"
    elif delta > 0.02:
        v = f"complete_playbook_exemplars_improved_cellrecall_delta_{delta}_small_n_{n_runs}"
    elif delta < -0.02:
        v = f"complete_playbook_exemplars_hurt_cellrecall_delta_{delta}_small_n_{n_runs}"
    else:
        v = f"complete_playbook_exemplars_inconclusive_cellrecall_delta_{delta}_small_n_{n_runs}"
    return v, delta, floored, outlier_fragile


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
    verdict, delta, floored, outlier_fragile = _verdict(control, treatment, rows, len(rows))
    acc_delta = None
    if (
        control["mean_reproduction_accuracy"] is not None
        and treatment["mean_reproduction_accuracy"] is not None
    ):
        acc_delta = round(
            treatment["mean_reproduction_accuracy"] - control["mean_reproduction_accuracy"], 4
        )

    base.update(
        {
            "honest_verdict": verdict,
            "skipped": skipped,
            "n_runs": len(rows),
            "control_exemplars_off": control,
            "treatment_exemplars_on": treatment,
            "playbook_exemplars_delta_accuracy": acc_delta,
            "playbook_exemplars_delta_cell_recall": delta,
            "metric_floored": bool(floored),
            "outlier_fragile_direction": bool(outlier_fragile),
            "rows": rows,
            "methodology_note": (
                "The Qwen3.5-9B proposer samples at temperature>0, so per-arm induction is "
                "non-deterministic; N = len(roster) x trials x 2 arms is small. Deltas are "
                "DIRECTIONAL, not significance claims. Both arms share the identical window, "
                "proposer config, and budget; only include_playbook_exemplars differs. Verified "
                "against the full winning trajectory (held-out beyond the k-window). Exact-grid-"
                "match accuracy floors at 0 for single-shot first-contact induction on these hard "
                "games, so the graded cell_recall is the discriminating metric; metric_floored=true "
                "means even that is at its floor and the offline metric cannot detect an effect "
                "(the same floor the AUTO_HUD_MASK levels_gained A/B hit) -- NOT evidence the "
                "feature helps or hurts. A live-submission levels_gained A/B is the better test."
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
