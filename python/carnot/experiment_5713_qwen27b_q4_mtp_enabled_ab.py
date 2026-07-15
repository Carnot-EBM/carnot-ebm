"""Experiment 5713: does enabling MTP (self-speculative decoding) change Qwen3.6-27B-MTP-GGUF's
(Q4_K_M, Q8 KV-cache) reinduction reliability, versus exp5599's non-MTP result for the SAME
weights/quantization/KV-cache combination? (task 14 continuation, operator-directed: "let's try
Qwen3.6-27B 4bit quant one last time with a Q8 kv-cache.")

**Precondition check found the requested config already measured.** exp5599
(`results/experiment_5599_reinduction_ab_lp85_levelup.json`) already ran EXACTLY Q4_K_M
`Qwen3.6-27B-MTP-GGUF` with `kv_quant="q8_0"`, at n=3 on the real reinduction path -- not a
loading failure or a degenerate n=1, a clean 3-draw measurement: `plan_rate_given_levelup=0/3`,
`mean_reinduce_duration_s=401.0` (294.9s/476.4s/431.2s), `heldout_accuracy` 0.333/0.0/0.333 (real
signal, never crossing the 1.0 acceptance threshold). Re-running that identical configuration
would be a doomed rerun per CLAUDE.md's Failed-Experiment Rerun Discipline -- no new variable, no
new information expected.

**The one genuinely untested variable for this weights/quant/KV combination: `mtp=False` was set
for exp5599's `candidate_27b` arm despite the model being named `Qwen3.6-27B-MTP-GGUF` (the 9B
baseline arm used `mtp=True`).** No inline rationale is recorded in exp5599's source for that
choice. `LocalGGUFProposer`'s MTP path is self-speculative (`--spec-type draft-mtp
--model-draft <same-path>` -- the SAME GGUF file serves as both target and draft, no separate
draft-model file needed), so nothing structurally should block enabling it for this model. This
experiment isolates exactly that one variable: same weights (Q4_K_M), same KV-cache precision
(Q8_0), same hardware (GPU 1, RTX 3090, this project's own CUDA build -- NOT the third-party
PrismML fork exp5709 needed), same task/methodology (real post-level-up `lp85` transitions,
`execute_bounded_llm_reinduction`) -- ONLY `mtp` flips `False -> True`.

Given the operator's explicit sample-size-fairness lesson from exp5709's n=1->n=3 upgrade this
same day, this experiment runs `n_repeats=3` from the start -- matching exp5599's own sample size
so the comparison is apples-to-apples without a follow-up correction needed.

**RESOLUTION: MTP structurally cannot run for this model on this hardware -- a hard OOM, not a
quality question.** The first launch attempt (background `n_repeats=3` run) stalled: the driver
process dropped to near-zero CPU and the launched `llama-server` subprocess was found `<defunct>`
(crashed, zombie) within the first health-check poll window -- but `LocalGGUFProposer._ensure_server()`
redirects the subprocess's stdout/stderr to `DEVNULL`, so the crash reason was invisible and the
driver would have polled a dead server for up to `load_wait_attempts` (600 x 2s = 20 minutes) per
repeat before giving up, for no new information. Killed and diagnosed directly instead: a manual
launch with visible output showed the target model (Q4_K_M, ~15.9GiB on disk) loaded fine, but
loading the DRAFT model -- self-speculative MTP loads the SAME GGUF file a SECOND time as a
separate CUDA buffer -- failed: `cudaMalloc failed: out of memory` trying to allocate ~15.6GiB on
top of the already-loaded target. Total demand (~32.6GB) exceeds the single RTX 3090's 24GB
outright. This is the root cause exp5599 almost certainly hit too (undocumented at the time,
hence `mtp=False` with no recorded rationale) -- and it explains why MTP works for the 9B arm
(9B x 2 copies comfortably fits 24GB) but not for a 27B-class model on a single card. The
precondition check now computes this directly from the real on-disk file size (2x file size vs
free VRAM) so this experiment blocks in under a second with the concrete numbers, rather than
burning 20 minutes x 3 repeats confirming a deterministic, instantly-reproducible failure.

Spec refs: REQ-ARC-WMTE-5599-4 (extends REQ-ARC-WMTE-5599's Q4 candidate measurement with the
one remaining untested serving-flag variable).
"""

from __future__ import annotations

import hashlib
import json
import os
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

EXPERIMENT_ID = "experiment_5713_qwen27b_q4_mtp_enabled_ab"
RESULT_RELATIVE_PATH = "results/experiment_5713_qwen27b_q4_mtp_enabled_ab.json"
SCHEMA = "carnot.exp5713.qwen27b_q4_mtp_enabled_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5713
GAME = "lp85"
N_REPEATS = 3  # matches exp5599's own sample size from the start -- no n=1 correction needed
EXPLORE_BUDGET = 6
TOTAL_BUDGET = 40
REINDUCTION_N_CTX = 22000
CANDIDATE_PORT = 8942  # distinct from exp5599's 8940/8941 -- avoid any stale-server collision
WEIGHT_PRECISION = "q4_k_m"
KV_CACHE_PRECISION = "q8_0"

EXP5599_HISTORICAL = {
    "current_9b_q4": {
        "experiment_id": 5599,
        "model": "Qwen3.5-9B-MTP-GGUF (Q4_K_M, MTP, GPU 1 CUDA)",
        "plan_rate_given_levelup": 1.0 / 3.0,
        "mean_reinduce_duration_s": 55.0,
    },
    "candidate_27b_q4_no_mtp": {
        "experiment_id": 5599,
        "model": "Qwen3.6-27B-MTP-GGUF (Q4_K_M, MTP DISABLED, Q8 KV-cache, GPU 1 CUDA)",
        "plan_rate_given_levelup": 0.0 / 3.0,
        "mean_reinduce_duration_s": 401.0,
        "note": "the config this experiment isolates ONE variable from -- same weights/quant/KV, "
        "MTP was off here",
    },
}

MANUAL_DIAGNOSTIC_CRASH_LOG_EXCERPT = (
    "0.02.081.619 I srv    load_model: [spec] estimated memory usage of draft model is "
    "15827.30 MiB\n"
    "0.03.665.224 W common_fit_params: failed to fit params to free device memory: "
    "n_gpu_layers already set by user to 999, abort\n"
    "0.12.397.645 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15621.78 MiB on "
    "device 0: cudaMalloc failed: out of memory\n"
    "0.12.397.652 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 16380622848\n"
    "0.12.486.968 E llama_model_load: error loading model: unable to allocate CUDA0 buffer\n"
    "0.12.486.979 E srv    load_model: failed to load draft model"
)

MODEL_SPECS = [
    {
        "name": "Qwen3.6-27B-MTP-Q4_K_M-MTP-enabled",
        "hf_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "role": "SAME weights/quantization (Q4_K_M) and KV-cache precision (Q8_0) exp5599 already "
        "measured with MTP disabled -- this run flips ONLY mtp=False -> True, the one untested "
        "serving-flag variable for this specific weights/quant/KV combination",
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
    "mtp_enabled",
    "serving_hardware",
    "per_draw_results",
    "arm_summary",
    "exp5599_historical_reference",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; the ONLY variable changed from exp5599's already-clean "
        "candidate_27b_q4_no_mtp measurement is mtp -- a genuinely isolated single-variable "
        "comparison, unlike exp5705/exp5709 which changed model family, quantization, AND "
        "serving stack simultaneously"
    },
    "mtp_enabled": {
        "principle": "the single independent variable this experiment isolates -- exp5599 set "
        "mtp=False for this model with no recorded rationale despite the model being named "
        "*-MTP-GGUF; this field makes the comparison's one difference explicit and auditable"
    },
    "weight_precision": {
        "principle": "held constant at exp5599's Q4_K_M -- NOT re-testing quantization level, "
        "only the MTP flag"
    },
    "kv_cache_precision": {
        "principle": "held constant at exp5599's Q8_0 KV-cache -- this is the exact config the "
        "operator asked to try; it was already measured (MTP off), so this field documents the "
        "held-constant baseline this run isolates one variable from"
    },
    "exp5599_historical_reference": {
        "principle": "names the prior no-MTP measurement this compares against and discloses it "
        "is cross-session, not re-measured fresh in this same run"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def _gpu1_free_mb() -> int:  # pragma: no cover - nvidia-smi hardware boundary
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[1]) if len(lines) > 1 else -1
    except Exception:
        return -1


def preconditions(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - live preflight
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
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    checks["qwen27b_mtp_gguf_cached"] = (
        any("qwen3.6-27b-mtp" in p.name.lower() for p in hub.glob("models--*"))
        if hub.exists()
        else False
    )
    checks["llama_server_cuda_binary_present"] = (
        Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server"
    ).exists()
    # Self-speculative MTP (--spec-type draft-mtp --model-draft <same-path>) loads the GGUF file
    # TWICE (target + draft), even though it's the same weights -- llama.cpp allocates two
    # separate CUDA buffers. A single-copy free-VRAM check is insufficient for this arm; compute
    # the real dual-load requirement from the actual on-disk file size (found empirically via a
    # direct manual server launch after the background run stalled: the target model loaded fine,
    # then the draft model's allocation failed with "cudaMalloc failed: out of memory" trying to
    # reserve ~15.6GiB on top of the already-loaded target).
    gguf_path = next(
        hub.glob("models--*Qwen3.6-27B-MTP*/snapshots/*/Qwen3.6-27B-Q4_K_M.gguf"), None
    )
    file_size_mb = (gguf_path.stat().st_size / (1024 * 1024)) if gguf_path else 0.0
    checks["mtp_dual_load_estimated_mb"] = round(2 * file_size_mb, 1)  # target + draft, same file
    checks["gpu1_free_mb"] = float(_gpu1_free_mb())
    checks["gpu1_free_vram_sufficient_for_mtp_dual_load"] = (
        file_size_mb > 0 and _gpu1_free_mb() >= 2 * file_size_mb
    )
    checks["ok"] = all(
        v for k, v in checks.items() if k not in ("mtp_dual_load_estimated_mb", "gpu1_free_mb")
    )
    return checks


_DIAGNOSTIC_ONLY_PRECONDITION_KEYS = ("ok", "mtp_dual_load_estimated_mb", "gpu1_free_mb")


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key in _DIAGNOSTIC_ONLY_PRECONDITION_KEYS:
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _wait_for_port_down(
    port: int, *, timeout_s: float = 30.0
) -> None:  # pragma: no cover - live server boundary
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


def _make_mtp_proposer() -> Any:  # pragma: no cover - live llama.cpp server boundary
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = "1"  # same opt-in 3090 route exp5599 used
    return LocalGGUFProposer(
        repo_substr="Qwen3.6-27B-MTP",
        port=CANDIDATE_PORT,
        mtp=True,  # the ONE variable flipped from exp5599's candidate_27b_q4_no_mtp arm
        kv_quant="q8_0",
        max_tokens=2560,
        n_ctx=REINDUCTION_N_CTX,
        timeout=1200,
    )


def _run_one_draw(*, proposer: Any, repeat: int) -> JsonDict:  # pragma: no cover - live ARC/LLM
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    row: JsonDict = {"arm": "qwen27b_q4_mtp_enabled", "repeat": repeat}

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
        {
            k: v
            for k, v in r.items()
            if k in ("round", "action", "proposer_ok", "message", "skipped")
        }
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
            "kv_cache_precision": KV_CACHE_PRECISION,
            "mtp_enabled": True,
            "serving_hardware": "nvidia_rtx_3090_gpu1_cuda",
            "per_draw_results": [],
            "arm_summary": {},
            "exp5599_historical_reference": EXP5599_HISTORICAL,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
            "manual_diagnostic_crash_confirmation": MANUAL_DIAGNOSTIC_CRASH_LOG_EXCERPT,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    proposer = _make_mtp_proposer()
    rows: list[JsonDict] = []
    try:
        for repeat in range(n_repeats):
            try:
                rows.append(_run_one_draw(proposer=proposer, repeat=repeat))
            except Exception as exc:
                rows.append(
                    {"arm": "qwen27b_q4_mtp_enabled", "repeat": repeat, "error": repr(exc)[:200]}
                )
    finally:
        proposer.stop()
        _wait_for_port_down(CANDIDATE_PORT)

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
                sum(
                    r["reinduce_duration_s"]
                    for r in levelup_rows
                    if r.get("reinduce_duration_s") is not None
                )
                / max(1, sum(1 for r in levelup_rows if r.get("reinduce_duration_s") is not None)),
                3,
            )
            if any(r.get("reinduce_duration_s") is not None for r in levelup_rows)
            else None
        ),
        "mean_heldout_accuracy": (
            round(
                sum(
                    r["heldout_accuracy"]
                    for r in levelup_rows
                    if r.get("heldout_accuracy") is not None
                )
                / max(1, sum(1 for r in levelup_rows if r.get("heldout_accuracy") is not None)),
                4,
            )
            if any(r.get("heldout_accuracy") is not None for r in levelup_rows)
            else None
        ),
    }

    if not levelup_rows:
        verdict = "complete: qwen27b_mtp_lp85_never_leveled_up_inconclusive"
    else:
        rate = arm_summary["plan_rate_given_levelup"]
        current_rate = EXP5599_HISTORICAL["current_9b_q4"]["plan_rate_given_levelup"]
        no_mtp_rate = EXP5599_HISTORICAL["candidate_27b_q4_no_mtp"]["plan_rate_given_levelup"]
        if rate is None:  # pragma: no cover - defensive; numeric when levelup_rows is non-empty
            verdict = "complete: qwen27b_mtp_result_inconclusive_vs_current_9b_baseline"
        elif rate > current_rate:
            verdict = "complete: qwen27b_mtp_plans_more_reliably_than_current_9b"
        elif rate == current_rate:
            verdict = "complete: qwen27b_mtp_ties_current_9b_plan_rate"
        elif rate > no_mtp_rate:
            verdict = "complete: qwen27b_mtp_improves_on_no_mtp_but_below_current_9b"
        else:
            verdict = "complete: qwen27b_mtp_plans_less_reliably_than_current_9b_ties_no_mtp"

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
        "kv_cache_precision": KV_CACHE_PRECISION,
        "mtp_enabled": True,
        "serving_hardware": "nvidia_rtx_3090_gpu1_cuda",
        "per_draw_results": rows,
        "arm_summary": arm_summary,
        "exp5599_historical_reference": EXP5599_HISTORICAL,
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
