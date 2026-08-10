#!/usr/bin/env python3
"""REQ-ARC-WMTE-6248: Pinductor-style REx refinement A/B (population search vs linear lineage).

WHAT THIS MEASURES. At the SAME per-cell LLM-call budget (1 induce + 3 refinements), does
Pinductor-style search structure (UCB1 parent selection over the candidate tree + QBC
committee-disagreement-ordered counterexamples) produce a better final engine than the current
production shape (refine the latest candidate, mismatches in corpus order)? Metric: HELD-slice
`change_fidelity` of each arm's final pick, per game, paired. Full plan + prior-failure block:
docs/research-notes/pinductor-rex-refinement-plan-2026-08-09.md

OPERATOR AUTHORIZATION. The refinement axis carries a standing hold in ops/known-issues.md
("banked, not queued" -- reopening requires an operator decision). The 2026-08-09 operator
directive "Let's plan out and prepare Pinductor to be run" is that decision, for this one
experiment.

PRE-REGISTERED GATE. REx beats linear on HELD change_fidelity in >= 4 of 6 games AND pooled mean
paired delta > 0. Fail -> the Pinductor-style refinement variant is retired
(retire_if_same_verdict). Pass -> authorizes ONLY a default-OFF live-path wiring follow-up, not a
default flip. Secondary (reported, not gated): any candidate reaching HELD change_fidelity >= 0.5
(the live trust threshold).

ISOLATION. CARNOT_ARC_E3_DIR must point at a private scratch dir BEFORE launch (read once at
import time by arc_executable_world_model). This guard exists because exp6247 wrote into the
shared results/arc_e3 store and clobbered the conductor's accumulated engines (caught only
because the conductor happened to be stopped). CARNOT_ARC_REFACTOR_SHOW_ENGINE=1 is forced in
BOTH arms: exp5766's null came from the engine source never reaching the refactor prompt, and
this is the banked fix's first completed measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

if not os.environ.get("CARNOT_ARC_E3_DIR"):
    raise SystemExit(
        "set CARNOT_ARC_E3_DIR to a private scratch directory BEFORE launching this script -- "
        "it must never write to the shared results/arc_e3 store (exp6247 incident)."
    )
# Both arms run with the engine-visible refactor prompt: the banked exp5766 instrument fix.
os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = "1"

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_rex_refinement as rex  # noqa: E402

OUT = REPO / "results" / "experiment_6248_pinductor_rex_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6248_CHECKPOINT", "/tmp/carnot_exp6248_checkpoint.json")
)
ROSTER = ("ft09", "tr87", "cn04", "ar25", "ka59", "re86")
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
BUDGET = 4  # LLM calls per arm-cell: 1 induce + 3 refinements
SEED = 6248
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8940
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500
GATE_MIN_GAMES_IMPROVED = 4
TRUST_THRESHOLD = 0.5


def _load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}


def _save_checkpoint(done: dict) -> None:
    CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    return _compute_hud_mask_from_frame(frame)


def _logical_hud_mask(game: str, cell: int):
    frame_mask = _frame_hud_mask(game)
    if frame_mask is None:
        return None
    return e3.logical_hud_mask(frame_mask, cell)


def _store_file(game: str) -> Path:
    return Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"


def _scorer(valid, hud_mask):
    """VALID-slice scorer for selection/QBC. Never sees the HELD slice."""

    def score_candidate(source: str) -> dict:
        try:
            engine = rex.load_engine_from_source(source, tag="score")
        except Exception as exc:  # noqa: BLE001
            return {
                "valid_fidelity": 0.0,
                "mismatches": [],
                "valid_accuracy": 0.0,
                "valid_n": 0,
                "valid_n_correct": 0,
                "load_error": repr(exc)[:160],
            }
        # hud_mask_enabled explicit per the REQ-ARC-WMTE-6233/6246 forced-mask convention.
        vr = e3.WorldModelVerifier(valid, hud_mask=hud_mask, hud_mask_enabled=True).score(
            engine, max_mismatch=len(valid)
        )
        return {
            "valid_fidelity": float(vr.change_fidelity),
            "mismatches": list(vr.mismatches),
            "valid_accuracy": float(vr.accuracy),
            "valid_n": int(vr.n),
            "valid_n_correct": int(vr.n_correct),
        }

    return score_candidate


def _make_verify_result(node: rex.RexNode, ordered_mismatches: list[dict]):
    """Build the VerifyResult `proposer.refactor` renders into the prompt.

    Only the fields `refactor_prompt` reads matter: n, n_correct, accuracy, mismatches.
    The mismatch ORDER is the QBC lever -- the prompt bounds to the first 5.
    """
    return e3.VerifyResult(
        n=node.valid_n,
        n_correct=node.valid_n_correct,
        accuracy=node.valid_accuracy,
        mismatches=list(ordered_mismatches),
    )


def _score_held(source: str, held, hud_mask) -> dict:
    try:
        engine = rex.load_engine_from_source(source, tag="held")
    except Exception as exc:  # noqa: BLE001
        return {"held_change_fidelity": 0.0, "held_load_error": repr(exc)[:160]}
    vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
    return {
        "held_change_fidelity": round(float(vr.change_fidelity), 4),
        "held_accuracy": round(float(vr.accuracy), 4),
        "held_cell_recall": round(float(vr.cell_recall), 4),
        "held_n_changing": int(vr.n_changing),
        "hud_mask_status": vr.hud_mask_status,
    }


def _run_cell(prop, game: str, arm: str, train, valid, held, hud_mask, cell: int) -> dict:
    store = _store_file(game)

    def read_store():
        return store.read_text() if store.exists() else None

    def write_store(text: str) -> None:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(text)

    t0 = time.time()
    summary = rex.run_rex(
        game,
        prop,
        train=train,
        valid=valid,
        cell=cell,
        budget=BUDGET,
        score_candidate=_scorer(valid, hud_mask),
        read_store_source=read_store,
        write_store_source=write_store,
        make_verify_result=_make_verify_result,
        use_ucb1=(arm == "rex"),
        use_qbc=(arm == "rex"),
    )
    row: dict = {
        "game": game,
        "arm": arm,
        "llm_calls": summary["llm_calls"],
        "nodes": summary["nodes"],
        "events": summary["events"],
        "final_idx": summary["final_idx"],
        "final_valid_fidelity": summary["final_valid_fidelity"],
        "wall_s": round(time.time() - t0, 1),
        "generator_healthy_after": bool(prop._healthy()),
    }
    # Any-candidate trust-threshold marker: score EVERY node on HELD (cheap, CPU-only)
    # so a mid-tree candidate crossing 0.5 is visible even when the final pick differs.
    if summary["final_source"]:
        row.update(_score_held(summary["final_source"], held, hud_mask))
        per_node_held = []
        for src in summary["node_sources"]:
            per_node_held.append(_score_held(src, held, hud_mask).get("held_change_fidelity", 0.0))
        row["held_per_node"] = per_node_held
        row["held_best_any_candidate"] = max(per_node_held) if per_node_held else None
    else:
        row["held_best_any_candidate"] = None
    return row


def build_artifact() -> dict:
    t0 = time.time()
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = os.environ.get(
        "CARNOT_ARC_GENERATOR_CUDA_GPU", "1"
    )
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"

    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    prop = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=SERVER_PORT,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=INDUCE_TIMEOUT_S,
    )
    try:
        server_up = prop._ensure_server()
    except GeneratorCudaRequiredError as exc:
        return {"honest_verdict": f"complete_blocked_cuda_unavailable_{exc!r}"[:200]}
    if not server_up:
        return {"honest_verdict": "complete_blocked_cuda_server_failed_to_start"}

    done = _load_checkpoint()
    rows = list(done.get("rows", []))
    done_keys = {(r["game"], r["arm"]) for r in rows if r.get("final_idx") is not None}
    rows = [r for r in rows if (r["game"], r["arm"]) in done_keys]

    for game in ROSTER:
        try:
            trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            rows.append({"game": game, "arm": "collect_error", "error": repr(exc)[:200]})
            done["rows"] = rows
            _save_checkpoint(done)
            continue
        train = trans[: -(N_VALID + N_HELD)]
        valid = trans[-(N_VALID + N_HELD) : -N_HELD]
        held = trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)
        for arm in ("linear", "rex"):
            if (game, arm) in done_keys:
                continue
            prop._ensure_server()
            try:
                row = _run_cell(prop, game, arm, train, valid, held, hud_mask, cell)
            except Exception as exc:  # noqa: BLE001
                row = {"game": game, "arm": arm, "error": repr(exc)[:200]}
            rows.append(row)
            done["rows"] = rows
            _save_checkpoint(done)
            print(
                f"[exp6248] {game} {arm}: final_idx={row.get('final_idx')} "
                f"valid={row.get('final_valid_fidelity')} held={row.get('held_change_fidelity')} "
                f"calls={row.get('llm_calls')}",
                flush=True,
            )

    per_game: dict = {}
    for r in rows:
        if r.get("arm") in ("linear", "rex"):
            per_game.setdefault(r["game"], {})[r["arm"]] = r

    deltas = []
    n_improve = 0
    n_comparable = 0
    any_trust = False
    for game, arms in per_game.items():
        lin, rx = arms.get("linear"), arms.get("rex")
        if not lin or not rx:
            continue
        lf, rf = lin.get("held_change_fidelity"), rx.get("held_change_fidelity")
        if lf is None or rf is None:
            continue
        n_comparable += 1
        delta = rf - lf
        deltas.append({"game": game, "linear": lf, "rex": rf, "delta": round(delta, 4)})
        if delta > 0:
            n_improve += 1
        # The trust marker considers EVERY candidate either arm produced, not only
        # the finals -- `held_best_any_candidate` carries the per-cell max.
        for cell_row in (lin, rx):
            best = cell_row.get("held_best_any_candidate")
            if best is not None and best >= TRUST_THRESHOLD:
                any_trust = True

    pooled = round(sum(d["delta"] for d in deltas) / len(deltas), 4) if deltas else None
    gate_met = (
        n_comparable >= len(ROSTER) - 1
        and n_improve >= GATE_MIN_GAMES_IMPROVED
        and pooled is not None
        and pooled > 0
    )

    art = {
        "experiment": "experiment_6248_pinductor_rex_ab",
        "title": "Pinductor-style REx refinement A/B: UCB1+QBC population search vs linear lineage",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "budget_llm_calls_per_arm_cell": BUDGET,
        "n_collect": N_COLLECT,
        "n_valid": N_VALID,
        "n_held": N_HELD,
        "per_game_results": rows,
        "paired_deltas": deltas,
        "n_games_comparable": n_comparable,
        "n_games_rex_improved": n_improve,
        "pooled_mean_delta": pooled,
        "gate_min_games_improved": GATE_MIN_GAMES_IMPROVED,
        "gate_met": bool(gate_met),
        "any_candidate_reached_trust_threshold": bool(any_trust),
        "trust_threshold": TRUST_THRESHOLD,
        "engine_visible_refactor_prompt": True,
        "operator_authorization": (
            "2026-08-09 operator directive 'Let's plan out and prepare Pinductor to be run' -- "
            "the explicit operator decision the standing refinement-axis hold required."
        ),
        "prior_failures": [
            {
                "experiment_id": "exp5766",
                "verdict": "cegis_refinement_null",
                "addressed_by": "instrument defect (engine source never in prompt) fixed and forced ON in both arms here",
            },
            {
                "experiment_id": "exp6091",
                "verdict": "blocked_reaper_zero_valid_cells",
                "addressed_by": "per-cell checkpoint + retry; conductor-isolation launch option per 2026-08-09 reaper findings",
            },
            {
                "experiment_id": "exp5764",
                "verdict": "single_shot_heldout_0.378",
                "addressed_by": "this measures SEARCH STRUCTURE on top of the same generator, a question single-shot quality does not answer",
            },
        ],
        "retire_if_same_verdict": True,
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "held-out change_fidelity is an oracle-distinct graded dynamics score, not the "
            "executable win-condition oracle."
        ),
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "generator": "unsloth/gemma-4-31B-it-qat-GGUF",
            "quant": "UD-Q4_K_XL",
            "kv_cache_quant": "q8_0",
            "port": SERVER_PORT,
        },
        "random_seed": SEED,
    }
    if gate_met:
        art["honest_verdict"] = (
            f"complete_rex_gate_met_{n_improve}_of_{n_comparable}_games_pooled_delta_{pooled}"
        )
    elif n_comparable == 0:
        art["honest_verdict"] = "complete_blocked_zero_comparable_cells_not_a_lever_result"
    else:
        art["honest_verdict"] = (
            f"complete_rex_gate_not_met_{n_improve}_of_{n_comparable}_games_pooled_delta_{pooled}_variant_retired"
        )
    art["duration_s"] = round(time.time() - t0, 3)
    payload = {k: v for k, v in art.items() if k != "duration_s"}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print("verdict:", art.get("honest_verdict"))
    print("wrote", OUT)
    if art.get("honest_verdict"):
        CHECKPOINT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
