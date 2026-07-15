"""Experiment 5709: does `prism-ml/Ternary-Bonsai-27B-gguf` (a ternary/2-bit quantization of
Qwen3.6-27B, served on a real NVIDIA GPU via a third-party llama.cpp fork) plan more reliably on
Carnot's real reinduction path than exp5599's Q4-quantized Qwen3.6-27B candidate or exp5705's
Q8_0 Gemma-4-31B-it candidate did (task 14 continuation, operator-directed).

Context: the operator asked to check `https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf`
("Ternary Bonsai") on CUDA. It advertises ~1.71 bits/weight ternary quantization ({-1,0,+1}
weights, GGUF Q2_0_g128 packing) of Qwen3.6-27B, claiming ~95% of FP16 intelligence retained on
generic benchmarks -- but it requires a bespoke third-party fork (`github.com/PrismML-Eng/
llama.cpp`, branch `prism`) with custom low-bit kernels; standard llama.cpp cannot load its
`Q2_0_g128` tensor type.

**Pre-integration audit (disclosed, not skipped).** Before building/running anything, the fork
was cloned and inspected: a normal llama.cpp fork layout (ggml/src/tools), no curl-pipe-to-shell
or remote-code-eval patterns, 217 stars, actively released the same day. One genuine concern was
found: grepping `ggml-cuda`/`ggml-hip`/`ggml-metal` for the ternary type names (`Q2_0_g128`,
`TQ1_0`, `TQ2_0`, `PQ2_0`) turned up NO dedicated kernel files -- only generic CPU-side hits in
`ggml-quants.c`. This raised a real possibility that CUDA offload for the ternary tensors would
silently fall back to slow CPU dequant, or fail to load. `src/models/dspark.cpp` -- the file HF's
API auto-parser mistakenly summarized as the whole model's "architecture" (it is actually the
separate EAGLE-style speculative-decoding drafter, not the ternary trunk) -- confirmed the fork
does implement genuinely novel per-model C++ code, not just a thin marketing wrapper.

**Empirical resolution: the audit's CUDA-kernel concern did NOT materialize.** Built with
`cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` (RTX 3090, this project's own
compute capability), the fork compiled cleanly end-to-end (`ggml-cuda` target built, `llama-server`
linked). Loading `Ternary-Bonsai-27B-Q2_0.gguf` (7.17GB, the file the model card's own quickstart
names) on GPU 1 (`CUDA_VISIBLE_DEVICES=1` -- GPU 0 is reserved for the conductor's own generator
per CLAUDE.md's hardware-allocation rule) used 22.5GB of real GPU memory (not near-zero, which
would indicate CPU fallback) and a real `/completion` call ("The capital of France is" ->
"Paris. Paris is the largest city in France...") returned coherent, correct text at 67.5 tok/s
decode -- genuinely fast, real GPU-accelerated ternary inference. The earlier grep-based concern
was a false alarm: the kernel implementation exists somewhere this audit's naming-based search
missed (plausibly a generic templated dequant path), not a functional gap.

**What this experiment does NOT re-litigate:** it does not re-run vLLM-on-ROCm feasibility
(ruled out in exp5705 -- unrelated to this fork), and it does not attempt to reproduce the
model card's own "~95% of FP16 intelligence" claim (measured by the vendor on EvalScope+vLLM on
an H100, a different serving stack, on generic thinking-mode benchmarks -- not this project's
code-induction task). This experiment measures ONE thing directly: does Ternary Bonsai plan
reliably on Carnot's own real `execute_bounded_llm_reinduction` path, the same methodology
exp5599 and exp5705 used.

**Serving-stack disclosure (load-bearing, not incidental).** This proposer points at a server
process started OUTSIDE this script's lifecycle (`LocalGGUFProposer._ensure_server()` reuses an
already-healthy server rather than launching one itself -- verified via `_healthy()` before this
experiment runs) because the binary is a third-party fork
(`/tmp/prismml-llamacpp-audit/build-cuda/bin/llama-server`) this project's own
`_generator_server_and_env()` resolver does not know about. If that external server is not
already running and healthy on `TERNARY_BONSAI_PORT`, this experiment blocks rather than trying
to auto-launch the wrong (standard) binary against a GGUF it cannot parse.

Spec refs: REQ-ARC-WMTE-5599-2 (extends REQ-ARC-WMTE-5599 and REQ-ARC-WMTE-5599-2's cost/benefit
finding to a third precision/serving-stack point).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
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

EXPERIMENT_ID = "experiment_5709_ternary_bonsai_cuda_reinduction_ab"
RESULT_RELATIVE_PATH = "results/experiment_5709_ternary_bonsai_cuda_reinduction_ab.json"
SCHEMA = "carnot.exp5709.ternary_bonsai_cuda_reinduction_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5709
GAME = "lp85"
N_REPEATS = 1  # matches exp5705's n=1 pattern -- no incremental checkpointing, real GPU cost/repeat
EXPLORE_BUDGET = 6
TOTAL_BUDGET = 40
REINDUCTION_N_CTX = 22000  # matches exp5599/exp5705 -- lp85's 64x64 grid overflows the class default

TERNARY_BONSAI_GGUF_PATH = (
    Path.home() / ".cache" / "carnot-full-precision-gguf" / "Ternary-Bonsai-27B-Q2_0.gguf"
)
TERNARY_BONSAI_PORT = 8951
TERNARY_BONSAI_SERVER_BINARY = Path(
    "/tmp/prismml-llamacpp-audit/build-cuda/bin/llama-server"
)
WEIGHT_PRECISION = "ternary_q2_0_g128"  # ~1.71 bits/weight per the model card; {-1,0,+1} weights

EXP5599_HISTORICAL = {
    "current_9b_q4": {
        "experiment_id": 5599,
        "model": "Qwen3.5-9B-MTP-GGUF (Q4_K_M, MTP, GPU 1 CUDA)",
        "plan_rate_given_levelup": 1.0 / 3.0,
        "mean_reinduce_duration_s": 55.0,
    },
    "candidate_27b_q4": {
        "experiment_id": 5599,
        "model": "Qwen3.6-27B-MTP-GGUF (Q4_K_M, non-MTP fallback, GPU 1 CUDA)",
        "plan_rate_given_levelup": 0.0 / 3.0,
        "mean_reinduce_duration_s": 401.0,
    },
}
EXP5705_HISTORICAL = {
    "gemma_31b_q8_0": {
        "experiment_id": 5705,
        "model": "Gemma-4-31B-it-GGUF (Q8_0, AMD iGPU HIP)",
        "plan_rate_given_levelup": 0.0 / 1.0,
        "mean_reinduce_duration_s": 2408.163,
    },
}

MODEL_SPECS = [
    {
        "name": "Ternary-Bonsai-27B-Q2_0",
        "hf_id": "prism-ml/Ternary-Bonsai-27B-gguf",
        "role": "ternary (~1.71 bits/weight) quantization of Qwen3.6-27B, served via a third-party "
        "llama.cpp fork (github.com/PrismML-Eng/llama.cpp) on a real NVIDIA RTX 3090 (GPU 1, CUDA) "
        "-- the operator's direct follow-up after exp5705's Q8_0 iGPU result, testing whether a "
        "fundamentally different (much more aggressive) quantization scheme on genuinely fast "
        "hardware changes the outcome",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "game",
    "n_repeats",
    "reinduction_n_ctx",
    "weight_precision",
    "kv_cache_precision",
    "serving_hardware",
    "serving_stack_provenance",
    "per_draw_results",
    "arm_summary",
    "exp5599_historical_reference",
    "exp5705_historical_reference",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; the PRIMARY verdict compares Ternary Bonsai against the "
        "current 9B baseline (same task/methodology) -- model family, quantization scheme, AND "
        "serving stack all differ from both prior comparators, so this is a real but non-isolated "
        "measurement, not a controlled precision-only comparison"
    },
    "weight_precision": {
        "principle": "honestly names the actual quantization scheme served (~1.71 bits/weight "
        "ternary GGUF Q2_0_g128), the most aggressive of the three candidates measured across "
        "exp5599 (Q4_K_M), exp5705 (Q8_0), and this experiment"
    },
    "serving_hardware": {
        "principle": "discloses this runs on a real discrete GPU (RTX 3090, GPU 1) unlike exp5705's "
        "iGPU fallback -- removes the hardware confound exp5705 disclosed, but introduces a NEW one "
        "(serving_stack_provenance) instead"
    },
    "serving_stack_provenance": {
        "principle": "discloses that this run uses a third-party fork's llama-server binary, NOT "
        "this project's own HIP/CUDA build -- a genuine confound this project's own build cannot "
        "remove (standard llama.cpp cannot load the Q2_0_g128 tensor type at all), disclosed rather "
        "than silently treated as equivalent to every other experiment's serving stack"
    },
    "exp5599_historical_reference": {
        "principle": "names the prior Q4 measurement this compares against and discloses it is "
        "cross-session, not re-measured fresh in this same run"
    },
    "exp5705_historical_reference": {
        "principle": "names the prior Q8_0 measurement (same day, same operator directive thread) "
        "this compares against, disclosed as cross-session context"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def _server_healthy(port: int = TERNARY_BONSAI_PORT, *, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as r:
            return b"ok" in r.read()
    except Exception:
        return False


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
    checks["ternary_bonsai_gguf_present"] = TERNARY_BONSAI_GGUF_PATH.exists()
    checks["prismml_server_binary_present"] = TERNARY_BONSAI_SERVER_BINARY.exists()
    # Deliberately does NOT auto-launch: the external server must already be healthy, since this
    # project's own _generator_server_and_env() resolver does not know about the third-party
    # binary and would launch the WRONG (standard) llama-server against a GGUF it cannot parse.
    checks["ternary_bonsai_server_already_healthy"] = _server_healthy()
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


def _make_ternary_bonsai_proposer() -> Any:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr="Ternary-Bonsai-27B-Q2_0",
        model_path=str(TERNARY_BONSAI_GGUF_PATH),
        port=TERNARY_BONSAI_PORT,
        mtp=False,
        kv_quant=None,
        max_tokens=2560,
        n_ctx=REINDUCTION_N_CTX,
        timeout=1200,
    )


def _run_one_draw(*, proposer: Any, repeat: int) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    row: JsonDict = {"arm": "ternary_bonsai_27b_q2_0", "repeat": repeat}

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
    row["rounds"] = [
        {k: v for k, v in r.items() if k in ("round", "action", "proposer_ok", "message", "skipped")}
        for r in outcome.rounds
    ]
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
            "weight_precision": WEIGHT_PRECISION,
            "kv_cache_precision": "f16_default_unquantized",
            "serving_hardware": "nvidia_rtx_3090_gpu1_cuda",
            "serving_stack_provenance": "third_party_fork_prismml_eng_llama_cpp_branch_prism",
            "per_draw_results": [],
            "arm_summary": {},
            "exp5599_historical_reference": EXP5599_HISTORICAL,
            "exp5705_historical_reference": EXP5705_HISTORICAL,
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

    proposer = _make_ternary_bonsai_proposer()
    rows: list[JsonDict] = []
    try:
        for repeat in range(n_repeats):
            try:
                rows.append(_run_one_draw(proposer=proposer, repeat=repeat))
            except Exception as exc:
                rows.append(
                    {"arm": "ternary_bonsai_27b_q2_0", "repeat": repeat, "error": repr(exc)[:200]}
                )
    finally:
        # Does NOT terminate the externally-launched server (proposer._proc is None since
        # _ensure_server() reused an already-healthy server rather than launching one itself).
        proposer.stop()

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

    if not levelup_rows:
        verdict = "complete: ternary_bonsai_lp85_never_leveled_up_inconclusive"
    else:
        rate = arm_summary["plan_rate_given_levelup"]
        current_rate = EXP5599_HISTORICAL["current_9b_q4"]["plan_rate_given_levelup"]
        if rate is None:
            verdict = "complete: ternary_bonsai_result_inconclusive_vs_current_9b_baseline"
        elif rate > current_rate:
            verdict = "complete: ternary_bonsai_plans_more_reliably_than_current_9b"
        elif rate == current_rate:
            verdict = "complete: ternary_bonsai_ties_current_9b_plan_rate"
        else:
            verdict = "complete: ternary_bonsai_plans_less_reliably_than_current_9b"

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
        "weight_precision": WEIGHT_PRECISION,
        "kv_cache_precision": "f16_default_unquantized",
        "serving_hardware": "nvidia_rtx_3090_gpu1_cuda",
        "serving_stack_provenance": "third_party_fork_prismml_eng_llama_cpp_branch_prism",
        "per_draw_results": rows,
        "arm_summary": arm_summary,
        "exp5599_historical_reference": EXP5599_HISTORICAL,
        "exp5705_historical_reference": EXP5705_HISTORICAL,
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
