"""Experiment 5705: does the model the 3rd-place ARC-AGI-3 team ("forge") actually used --
Gemma-4-31B-it, full (non-quantized) precision -- plan more reliably on Carnot's real
reinduction path than exp5599's Q4-quantized Qwen3.6-27B candidate did (task 14,
operator-directed follow-up).

Context: exp5599 found `Qwen3.6-27B-MTP` (Q4_K_M GGUF, non-MTP fallback, served on a single
24GB RTX 3090) planned LESS reliably (0/3 vs the current 9B's 1/3) and took ~7x longer
(~401s vs ~55s mean reinduce duration) on the REAL `execute_bounded_llm_reinduction` path
against post-level-up `lp85` transitions. The operator asked why this contradicted forge's
real success running Gemma-4-31B-it at full precision via vLLM on a 96GB RTX Pro 6000 --
multiple confounds were identified (different model, different quantization, different
serving stack, different hardware, n=3), and the operator asked to isolate what's tractable
on this box: "we have plenty of VRAM on our AMD iGPU if we want to try the full model
weights instead of 4bit quants and/or full kv-cache key size."

vLLM itself was found infeasible on this hardware (the PyPI `vllm` wheel is CUDA-only;
vLLM's ROCm support has no PyPI distribution and has historically targeted MI-series
datacenter cards, not this consumer gfx1150 iGPU). The FIRST candidate tried for the
full-precision arm was `unsloth/Qwen3.6-27B` (same model family as exp5599's Q4 candidate,
for a precision-only isolation) -- this was ABANDONED after three real, reproducible load
failures on the HIP-built llama.cpp binary (`~/.cache/llama.cpp-master/build-hip/bin/
llama-server`): the default `-fit` heuristic hard-hung (zero I/O progress for 12+ minutes,
confirmed via `/proc/PID/io`); `-fit off` hard-stalled again at a later step (zero RSS/IO
progress for 5+ minutes, one thread pinned near 100% CPU, no backtrace tooling available on
this box -- `ptrace` restricted, no `perf`, no root); `-fit off --parallel 1` crawled at
~11MB/s (over an hour to finish loading). All three failures are consistent with this
specific HIP build mishandling Qwen3.6's unusual hybrid `Qwen3_5ForConditionalGeneration`
architecture (mixed "linear_attention"/"full_attention" layers, a newer, less-common
pattern) -- NOT a flag-tuning problem, a real compatibility gap, reported to and confirmed
by the operator rather than guessed past indefinitely.

**Operator-directed pivot:** switch the full-precision candidate to `google/gemma-4-31B-it`
(62.6GB, native BF16, `Gemma4ForConditionalGeneration`) -- the model forge ACTUALLY used, and
a conventional sliding-window + full-attention architecture (the same mature pattern Gemma
2/3 already used, well-supported in llama.cpp) rather than Qwen3.6's newer hybrid. This
directly answers the operator's original question (why did forge succeed with a bigger model
when our test showed a bigger model losing) with the ACTUAL model in question, at full
precision, on Carnot's own real task -- a MORE informative test than the originally-planned
same-model precision isolation, at the honest cost of no longer isolating precision as a
single variable against exp5599's Q4 Qwen candidate (model family changes too). Converted to
a NON-QUANTIZED BF16 GGUF via `convert_hf_to_gguf.py --outtype bf16`, served through the SAME
HIP-built llama.cpp binary every other local-GGUF experiment in this project uses, with full
(non-quantized) F16 KV cache (`kv_quant=None`, llama.cpp's own default) -- directly addressing
the "full kv-cache key size" half of the operator's request. `-fit off` is passed proactively
(learned from the Qwen3.6 diagnostic loop) even though this conventional architecture is
expected not to need it.

HONEST DISCLOSURE (not hidden): this does NOT reproduce exp5599's Q4 Qwen candidate_27b arm
or the frozen current-9B arm fresh in this same run -- it cites their historical numbers
(different session, different date, AND now a different model for the Q4 comparator) as
context, not as a controlled precision-only baseline. The comparison this experiment DOES
make cleanly is: does Gemma-4-31B-it (full precision, the leader's real model) plan
competitively against the current 9B's historical plan rate on the SAME real task -- that
is directly informative regardless of the Qwen-specific quantization question exp5599 raised.

Spec refs: REQ-ARC-WMTE-5599-2 (extends REQ-ARC-WMTE-5599's cost/benefit finding).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5705_full_precision_27b_vs_4bit_quant_ab"
RESULT_RELATIVE_PATH = "results/experiment_5705_full_precision_27b_vs_4bit_quant_ab.json"
SCHEMA = "carnot.exp5705.full_precision_27b_vs_4bit_quant_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5705
GAME = "lp85"
N_REPEATS = 3
EXPLORE_BUDGET = 6
TOTAL_BUDGET = 40
REINDUCTION_N_CTX = 22000  # matches exp5599 -- lp85's 64x64 grid overflows the class default
FULL_PRECISION_GGUF_PATH = (
    Path.home() / ".cache" / "carnot-full-precision-gguf" / "gemma-4-31B-it-BF16.gguf"
)
FULL_PRECISION_PORT = 8950

# Historical reference numbers from exp5599 (results/experiment_5599_reinduction_ab_lp85_levelup.json,
# 2026-07-13) -- NOT re-measured here; cited as CONTEXT, not a controlled precision-only baseline (the
# candidate_27b_q4 row is now a DIFFERENT model -- Qwen3.6-27B -- from this experiment's Gemma-4-31B-it
# full-precision arm; see this module's docstring for why the candidate was switched).
EXP5599_HISTORICAL = {
    "current_9b_q4": {
        "experiment_id": 5599,
        "model": "Qwen3.5-9B-MTP-GGUF (Q4_K_M, MTP, GPU 1 CUDA)",
        "plan_rate_given_levelup": 1.0 / 3.0,
        "mean_reinduce_duration_s": 55.0,
    },
    "candidate_27b_q4": {
        "experiment_id": 5599,
        "model": "Qwen3.6-27B-MTP-GGUF (Q4_K_M, non-MTP fallback, GPU 1 CUDA) -- a DIFFERENT model "
        "family from this experiment's full-precision Gemma-4-31B-it candidate; not a precision-only "
        "comparator for this run",
        "plan_rate_given_levelup": 0.0 / 3.0,
        "mean_reinduce_duration_s": 401.0,
    },
}

MODEL_SPECS = [
    {
        "name": "Gemma-4-31B-it-BF16",
        "hf_id": "google/gemma-4-31B-it",
        "role": "full-precision (non-quantized) candidate -- the model forge (3rd-place ARC-AGI-3) "
        "actually used, converted to BF16 GGUF locally after Qwen3.6-27B was abandoned (3 real load "
        "failures on this hardware's HIP llama.cpp build, unrelated to precision)",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "game",
    "n_repeats",
    "reinduction_n_ctx",
    "kv_cache_precision",
    "serving_hardware",
    "per_draw_results",
    "arm_summary",
    "exp5599_historical_reference",
    "qwen_q4_context_comparison",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; the PRIMARY verdict compares full-precision Gemma-4-31B-it "
        "against the current 9B baseline (same task/methodology) -- NOT against exp5599's Q4 Qwen "
        "candidate, since that comparison changed both model family AND precision together"
    },
    "kv_cache_precision": {
        "principle": "documents that this run uses llama.cpp's default full-precision (f16) KV "
        "cache, not exp5599's q8_0-quantized cache -- directly addresses the operator's "
        "'full kv-cache key size' request"
    },
    "serving_hardware": {
        "principle": "honestly discloses the NECESSARY hardware confound: a 62.6GB BF16 model "
        "cannot fit on the single 24GB RTX 3090 exp5599 used, so this arm runs on the AMD iGPU "
        "instead -- serving stack (llama.cpp) is held constant, hardware is not"
    },
    "exp5599_historical_reference": {
        "principle": "CLAUDE.md Failed-Experiment Rerun Discipline analog -- names the prior "
        "measurement this compares against and discloses it is cross-session, not re-measured "
        "fresh in this same run"
    },
    "qwen_q4_context_comparison": {
        "principle": "explicitly labeled CONTEXT ONLY, never the primary verdict -- comparing "
        "Gemma-4-31B-it (full precision) against exp5599's Qwen3.6-27B (Q4) conflates model "
        "family and precision; reporting it as a clean isolation would be dishonest"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(GAME, scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: F401
        from carnot.agentic.arc_llm_reinduction import (  # noqa: F401
            execute_bounded_llm_reinduction,
        )

        checks["reinduction_import"] = True
    except Exception:
        checks["reinduction_import"] = False
    checks["full_precision_gguf_present"] = FULL_PRECISION_GGUF_PATH.exists()
    checks["hip_llama_server_binary_present"] = (
        Path.home() / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server"
    ).exists()
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _wait_for_port_down(port: int, *, timeout_s: float = 30.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                time.sleep(1)
                continue
        except Exception:
            return
    time.sleep(2)


def _make_full_precision_proposer() -> Any:
    import os

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # Explicitly unset the 3090-pinning env var so _generator_server_and_env() routes to the
    # default iGPU HIP build (the 62.6GB weights do not fit a single 24GB 3090).
    os.environ.pop("CARNOT_ARC_GENERATOR_CUDA_GPU", None)
    return LocalGGUFProposer(
        repo_substr="gemma-4-31B-it-BF16",
        model_path=str(FULL_PRECISION_GGUF_PATH),
        port=FULL_PRECISION_PORT,
        mtp=False,
        kv_quant=None,  # full-precision (f16) KV cache -- llama.cpp's own default
        max_tokens=2560,
        n_ctx=REINDUCTION_N_CTX,
        timeout=1200,  # a 62.6GB BF16 model on the iGPU may be materially slower than the Q4 27B
        extra_server_args=("-fit", "off"),  # proactive, learned from the Qwen3.6 diagnostic loop
    )


def _run_one_draw(*, proposer: Any, repeat: int) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    row: JsonDict = {"arm": "full_precision_27b_bf16", "repeat": repeat}

    policy = E3AgentPolicy(GAME, proposer=proposer, explore_budget=EXPLORE_BUDGET)
    lb.run_game(GAME, policy, budget=TOTAL_BUDGET)
    all_transitions = list(policy.transitions)
    row["transition_count"] = len(all_transitions)

    first_levelup_index = next(
        (i for i, t in enumerate(all_transitions) if t.level_after > t.level_before), None
    )
    if first_levelup_index is None:
        row["levelup_reached"] = False
        return row
    row["levelup_reached"] = True
    row["actions_to_levelup"] = int(first_levelup_index) + 1
    induce_transitions = all_transitions[: first_levelup_index + 2]
    row["induce_transition_count"] = len(induce_transitions)

    reinduce_started = time.time()
    outcome = execute_bounded_llm_reinduction(
        game=policy.short,
        transitions=induce_transitions,
        cell=policy.cell,
        root_grid=policy.root_grid,
        proposer=proposer,
        candidate_provider=policy._world_model_candidates,
        load_engine=e3.load_engine,
        plan_in_model=e3.plan_in_model,
    )
    row["reinduce_duration_s"] = round(time.time() - reinduce_started, 3)
    row["planned"] = bool(outcome.planned)
    row["skipped"] = outcome.skipped
    row["plan_length"] = len(outcome.plan)
    row["heldout_accuracy"] = outcome.heldout_accuracy
    row["refinement_rounds_used"] = int(outcome.refinement_rounds_used)
    return row


def build_artifact(*, n_repeats: int = N_REPEATS, root: Path = REPO_ROOT) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "game": GAME,
            "n_repeats": int(n_repeats),
            "reinduction_n_ctx": REINDUCTION_N_CTX,
            "kv_cache_precision": "f16_default_unquantized",
            "serving_hardware": "amd_strix_point_gfx1150_igpu_rocm_hip",
            "per_draw_results": [],
            "arm_summary": {},
            "exp5599_historical_reference": EXP5599_HISTORICAL,
            "qwen_q4_context_comparison": "not_applicable_blocked",
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    proposer = _make_full_precision_proposer()
    rows: list[JsonDict] = []
    try:
        for repeat in range(n_repeats):
            try:
                rows.append(_run_one_draw(proposer=proposer, repeat=repeat))
            except Exception as exc:
                rows.append({"arm": "full_precision_27b_bf16", "repeat": repeat, "error": repr(exc)[:200]})
    finally:
        proposer.stop()
        _wait_for_port_down(FULL_PRECISION_PORT)

    levelup_rows = [r for r in rows if r.get("levelup_reached")]
    planned_rows = [r for r in levelup_rows if r.get("planned")]
    arm_summary = {
        "n_attempted": len(rows),
        "n_levelup_reached": len(levelup_rows),
        "n_planned": len(planned_rows),
        "plan_rate_given_levelup": (
            round(len(planned_rows) / len(levelup_rows), 4) if levelup_rows else None
        ),
        "mean_reinduce_duration_s": (
            round(
                sum(r["reinduce_duration_s"] for r in levelup_rows if r.get("reinduce_duration_s") is not None)
                / max(1, sum(1 for r in levelup_rows if r.get("reinduce_duration_s") is not None)),
                3,
            )
            if any(r.get("reinduce_duration_s") is not None for r in levelup_rows)
            else None
        ),
        "mean_heldout_accuracy": (
            round(
                sum(r["heldout_accuracy"] for r in levelup_rows if r.get("heldout_accuracy") is not None)
                / max(1, sum(1 for r in levelup_rows if r.get("heldout_accuracy") is not None)),
                4,
            )
            if any(r.get("heldout_accuracy") is not None for r in levelup_rows)
            else None
        ),
    }

    # The CLEAN comparison this experiment supports: gemma-4-31B-it (full precision) vs the
    # current 9B, same task/methodology (though cross-session -- see module docstring). The Q4
    # Qwen candidate comparison is DISCLOSED CONTEXT ONLY (different model AND precision changed
    # together), never the basis for the primary verdict.
    qwen_q4_context_comparison = "not_applicable_never_leveled_up"
    if not levelup_rows:
        verdict = "complete: gemma_full_precision_lp85_never_leveled_up_inconclusive"
    else:
        fp_rate = arm_summary["plan_rate_given_levelup"]
        current_rate = EXP5599_HISTORICAL["current_9b_q4"]["plan_rate_given_levelup"]
        q4_rate = EXP5599_HISTORICAL["candidate_27b_q4"]["plan_rate_given_levelup"]
        if fp_rate is None:
            verdict = "complete: gemma_full_precision_result_inconclusive_vs_current_9b_baseline"
        elif fp_rate > current_rate:
            verdict = "complete: gemma_full_precision_plans_more_reliably_than_current_9b"
        elif fp_rate == current_rate:
            verdict = "complete: gemma_full_precision_ties_current_9b_plan_rate"
        else:
            verdict = "complete: gemma_full_precision_plans_less_reliably_than_current_9b"
        if fp_rate is not None:
            if fp_rate > q4_rate:
                qwen_q4_context_comparison = (
                    "gemma_full_precision_beats_qwen_q4_context_only_different_model_and_precision"
                )
            elif fp_rate == q4_rate:
                qwen_q4_context_comparison = (
                    "gemma_full_precision_ties_qwen_q4_context_only_different_model_and_precision"
                )
            else:
                qwen_q4_context_comparison = (
                    "gemma_full_precision_below_qwen_q4_context_only_different_model_and_precision"
                )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "game": GAME,
        "n_repeats": int(n_repeats),
        "reinduction_n_ctx": REINDUCTION_N_CTX,
        "kv_cache_precision": "f16_default_unquantized",
        "serving_hardware": "amd_strix_point_gfx1150_igpu_rocm_hip",
        "per_draw_results": rows,
        "arm_summary": arm_summary,
        "exp5599_historical_reference": EXP5599_HISTORICAL,
        "qwen_q4_context_comparison": qwen_q4_context_comparison,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
