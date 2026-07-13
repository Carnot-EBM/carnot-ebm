"""Experiment 5597: does the second official larger candidate named by task 13 -- Qwen3.6-35B-
A3B-MTP (MoE, ~3B active params) -- improve tier-3 induction quality over the frozen live
generator (Qwen3.5-9B-MTP), following up on exp5596's dense-27B result?

Context: ops/known-issues.md task 13 (2026-07-12, HIGH PRIORITY), same investigation as
exp5596 (REQ-ARC-WMTE-5596). exp5596's dense Qwen3.6-27B-MTP candidate showed materially higher
heldout_accuracy than the current generator on both tested games (m0r0: 0.0->0.5, sk48:
0.2->1.0), while running WITHOUT MTP (self-draft OOM'd: 2x a 16.3GB file exceeds a single 24GB
RTX 3090). That result's own spec entry flagged the 35B MoE variant as "a natural follow-on if
this result is promising" -- this experiment is that follow-on.

MOE VRAM EXPECTATION: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`'s Q4_K_M quant is a LARGER file than
the dense 27B (35B total params vs 27B dense, even though only ~3B are active per token -- MoE
sparsity reduces COMPUTE per token, not the STORED weight size, which self-draft doubles
regardless of sparsity). This experiment reuses exp5596's exact two-step MTP feasibility check
(`_candidate_declares_mtp_metadata` + `_candidate_mtp_self_draft_fits_vram`, unmodified logic,
duplicated here per-candidate rather than shared, mirroring exp5594/5595/5596's established
one-file-per-experiment convention) -- so if self-draft doesn't fit (very likely, since the
27B's 16.3GB file already didn't fit and this file is larger), the candidate arm again correctly
falls back to non-MTP instead of crash-looping, exactly as exp5596 found necessary.

Both arms are pinned to the SAME hardware tier (GPU 1, one of the dev box's two RTX 3090s, the
outer loop's own allocated card per CLAUDE.md's GPU-allocation rule) via
CARNOT_ARC_GENERATOR_CUDA_GPU=1 and a fresh, non-default port per arm, with each arm's server
explicitly stopped (`proposer.stop()` + `_wait_for_port_down`) before the next starts -- the
same fix exp5596 required after finding a still-resident prior-arm server silently starves the
next arm's free-VRAM guard onto the slow iGPU.

SCOPE (honest): like exp5596, this measures induction QUALITY via the existing
WorldModelVerifier.heldout_accuracy on the same 2-game roster (m0r0, sk48), not a full
actions-to-first-win live-solve sweep.

Per the task's own guardrail (mirroring task 7's frozen-stack discipline): this is an OFFLINE DEV
MEASUREMENT, not a live-stack change. Do NOT flip the frozen live generator based on this
experiment alone; report the delta and require an explicit operator decision.

Spec refs: REQ-ARC-WMTE-5597, SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED,
SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA.
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

EXPERIMENT_ID = "experiment_5597_generator_size_ab_qwen35b_moe_vs_current"
RESULT_RELATIVE_PATH = "results/experiment_5597_generator_size_ab_qwen35b_moe_vs_current.json"
SCHEMA = "carnot.exp5597.generator_size_ab_qwen35b_moe_vs_current.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5597
DEFAULT_ROSTER = ("m0r0", "sk48")
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
CURRENT_REPO_SUBSTR = "Qwen3.5-9B-MTP"
CANDIDATE_REPO_SUBSTR = "Qwen3.6-35B-A3B-MTP"
SHARED_MAX_TOKENS = 2560
CANDIDATE_MIN_FREE_MB = 20000  # precondition floor; real feasibility is checked separately below

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "current frozen live-submission generator (arm=current)",
    },
    {
        "name": "Qwen3.6-35B-A3B-MTP",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "role": "candidate larger, MoE, MTP-capable generator -- exp5596's follow-on candidate "
        "(arm=candidate)",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "candidate_declares_mtp_metadata",
    "candidate_mtp_self_draft_fits_vram",
    "candidate_mtp_self_draft_detail",
    "candidate_mtp_used",
    "roster",
    "per_game_results",
    "current_induction_success_count",
    "candidate_induction_success_count",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a larger generator helping, not helping, or being "
        "untestable on this hardware are all distinct, real, citable outcomes"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- both arms invoke a real local GGUF proposer, not mocked"
    },
    "candidate_declares_mtp_metadata": {
        "principle": "checked via direct GGUF metadata inspection (nextn_predict_layers), not "
        "assumed -- architectural MTP support is a DIFFERENT fact from whether self-draft "
        "actually fits in available VRAM (see candidate_mtp_self_draft_fits_vram)"
    },
    "candidate_mtp_self_draft_fits_vram": {
        "principle": "self-draft MTP loads the SAME GGUF file twice (main + draft); exp5596 "
        "found this can OOM even when the metadata declares support, so this is checked "
        "separately before attempting a real launch, not assumed from metadata alone"
    },
    "candidate_mtp_used": {
        "principle": "the actual runtime decision (metadata support AND VRAM feasibility); "
        "this is what determines whether any wall-clock delta between arms is confounded by "
        "an MTP-vs-no-MTP asymmetry, not the metadata declaration alone"
    },
    "solve_provenance": {
        "principle": "development_proxy -- offline dev measurement per task 13's own guardrail; "
        "does not flip the frozen live-submission generator"
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
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_executable_world_model import (  # noqa: F401
            LocalGGUFProposer,
            WorldModelVerifier,
        )

        checks["e3_policy_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    checks["current_gguf_cached"] = (
        any(CURRENT_REPO_SUBSTR.lower() in p.name.lower() for p in hub.glob("models--*"))
        if hub.exists()
        else False
    )
    checks["candidate_gguf_cached"] = (
        any(CANDIDATE_REPO_SUBSTR.lower() in p.name.lower() for p in hub.glob("models--*"))
        if hub.exists()
        else False
    )
    checks["llama_server_binary_present"] = bool(
        list((Path.home() / ".cache").glob("llama.cpp*/build/bin/llama-server"))
    )
    checks["gpu1_free_vram_sufficient"] = _gpu1_free_mb() >= CANDIDATE_MIN_FREE_MB
    checks["ok"] = all(checks.values())
    return checks


def _gpu1_free_mb() -> int:
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


def _candidate_declares_mtp_metadata() -> bool:
    """Direct GGUF metadata check -- does the candidate declare an MTP self-draft head?"""

    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    path = _resolve_gguf(CANDIDATE_REPO_SUBSTR)
    if not path:
        return False
    try:
        with open(path, "rb") as f:
            # cheap textual scan of the leading metadata region for the nextn/draft key name --
            # avoids requiring the gguf package as a hard dependency for this check.
            header = f.read(1 << 20)
        return b"nextn_predict_layers" in header or b"n_predict_layers" in header
    except Exception:
        return False


def _candidate_mtp_self_draft_fits_vram() -> tuple[bool, str]:
    """Real hardware feasibility check (exp5596's fix, reused unmodified): llama.cpp's
    SELF-draft MTP loads the SAME GGUF file TWICE (main model + `--model-draft <same path>`),
    so the VRAM footprint is roughly 2x the file size plus KV cache and CUDA overhead --
    independent of MoE sparsity, since self-draft stores the full weight file regardless of how
    many experts are active per token. See exp5596's spec entry for the exact OOM this check was
    built to pre-empt on the dense 27B candidate."""

    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    path = _resolve_gguf(CANDIDATE_REPO_SUBSTR)
    if not path:
        return False, "candidate GGUF not resolvable"
    try:
        file_mb = Path(path).stat().st_size / (1024 * 1024)
    except Exception:
        return False, "could not stat candidate GGUF file size"
    self_draft_estimate_mb = 2.0 * file_mb  # main + draft, same file, before KV/overhead
    free_mb = _gpu1_free_mb()
    if free_mb < 0:
        return False, "GPU 1 free VRAM unreadable"
    fits = free_mb >= (self_draft_estimate_mb + 2000)  # +2GB margin for KV cache + CUDA overhead
    detail = (
        f"self-draft estimate {self_draft_estimate_mb:.0f}MB (2x {file_mb:.0f}MB file) vs "
        f"{free_mb}MB free on GPU 1 -- {'fits' if fits else 'does NOT fit'} with a 2000MB margin"
    )
    return fits, detail


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
    """Poll until a stopped llama-server's port stops answering /health (bounded), so the
    NEXT arm's GPU free-VRAM guard sees the reclaimed memory rather than a still-warm process."""

    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                time.sleep(1)
                continue
        except Exception:
            return  # port is down -- process has exited
    # timed out still responding; give the OS a last moment regardless
    time.sleep(2)


def _run_one_arm(
    game: str, *, arm: str, explore_budget: int, total_budget: int, candidate_mtp_used: bool
) -> JsonDict:
    import os

    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, WorldModelVerifier

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = "1"
    if arm == "current":
        proposer = LocalGGUFProposer(
            repo_substr=CURRENT_REPO_SUBSTR,
            port=8941,
            mtp=True,
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=SHARED_MAX_TOKENS,
        )
    else:
        # mtp driven by the empirically-verified runtime decision (metadata support AND VRAM
        # feasibility), not a hardcoded assumption -- see module docstring's "checked, not
        # assumed" principle, reusing exp5596's self-draft-OOM finding.
        proposer = LocalGGUFProposer(
            repo_substr=CANDIDATE_REPO_SUBSTR,
            port=8942,
            mtp=candidate_mtp_used,
            kv_quant="q8_0",
            max_tokens=SHARED_MAX_TOKENS,
        )

    # STOP the server after this arm, always (finally) -- both arms share GPU 1; a still-resident
    # server from a PRIOR arm consumes VRAM that starves the free-mem guard for the NEXT arm,
    # causing a silent fallback to the slow iGPU (exp5596's fix, reused here unmodified).
    try:
        policy = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
        lb.run_game(game, policy, budget=total_budget)
        active_transitions = list(policy.transitions)[:10]

        if not active_transitions:
            return {"game": game, "arm": arm, "transition_count": 0, "induction_ok": False}

        induce_started = time.time()
        ok, detail = proposer.induce(policy.short, active_transitions, policy.cell)
        induce_duration_s = round(time.time() - induce_started, 3)

        row: JsonDict = {
            "game": game,
            "arm": arm,
            "transition_count": len(active_transitions),
            "induction_ok": bool(ok),
            "induce_duration_s": induce_duration_s,
        }
        if not ok:
            row["induction_failure_detail"] = str(detail)[:300]
            return row

        engine, _is_level_complete = e3.load_engine(policy.short)
        verify_result = WorldModelVerifier(active_transitions).score(engine)
        row["heldout_accuracy"] = verify_result.accuracy
        row["cell_recall"] = verify_result.cell_recall
        return row
    finally:
        proposer.stop()
        _wait_for_port_down(proposer.port)  # let the OS reclaim VRAM before the next arm's guard


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    explore_budget: int = DEFAULT_EXPLORE_BUDGET,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
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
            "candidate_declares_mtp_metadata": False,
            "candidate_mtp_self_draft_fits_vram": False,
            "candidate_mtp_self_draft_detail": "",
            "candidate_mtp_used": False,
            "roster": list(roster),
            "per_game_results": [],
            "current_induction_success_count": 0,
            "candidate_induction_success_count": 0,
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

    candidate_declares_mtp = _candidate_declares_mtp_metadata()
    candidate_vram_fits, candidate_vram_detail = _candidate_mtp_self_draft_fits_vram()
    candidate_mtp_used = candidate_declares_mtp and candidate_vram_fits

    rows: list[JsonDict] = []
    for game in roster:
        for arm in ("current", "candidate"):
            try:
                rows.append(
                    _run_one_arm(
                        game,
                        arm=arm,
                        explore_budget=explore_budget,
                        total_budget=total_budget,
                        candidate_mtp_used=candidate_mtp_used,
                    )
                )
            except Exception as exc:
                rows.append({"game": game, "arm": arm, "error": repr(exc)[:200]})

    current_success = sum(1 for r in rows if r.get("arm") == "current" and r.get("induction_ok"))
    candidate_success = sum(
        1 for r in rows if r.get("arm") == "candidate" and r.get("induction_ok")
    )
    current_accuracies = [
        r["heldout_accuracy"] for r in rows if r.get("arm") == "current" and r.get("induction_ok")
    ]
    candidate_accuracies = [
        r["heldout_accuracy"] for r in rows if r.get("arm") == "candidate" and r.get("induction_ok")
    ]

    if current_success == 0 and candidate_success == 0:
        verdict = "complete: generator_size_ab_neither_arm_induced_inconclusive"
    elif candidate_success > current_success:
        verdict = (
            f"complete: generator_size_ab_candidate_more_reliable_"
            f"{current_success}_to_{candidate_success}_successes"
        )
    elif candidate_success < current_success:
        verdict = (
            f"complete: generator_size_ab_current_more_reliable_"
            f"{candidate_success}_to_{current_success}_successes"
        )
    elif (
        candidate_accuracies
        and current_accuracies
        and sum(candidate_accuracies) / len(candidate_accuracies)
        > sum(current_accuracies) / len(current_accuracies)
    ):
        verdict = "complete: generator_size_ab_equal_success_candidate_higher_accuracy"
    elif (
        candidate_accuracies
        and current_accuracies
        and sum(candidate_accuracies) / len(candidate_accuracies)
        < sum(current_accuracies) / len(current_accuracies)
    ):
        verdict = "complete: generator_size_ab_equal_success_current_higher_accuracy"
    else:
        verdict = "complete: generator_size_ab_honest_null_no_measured_difference"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "candidate_declares_mtp_metadata": candidate_declares_mtp,
        "candidate_mtp_self_draft_fits_vram": candidate_vram_fits,
        "candidate_mtp_self_draft_detail": candidate_vram_detail,
        "candidate_mtp_used": candidate_mtp_used,
        "roster": list(roster),
        "per_game_results": rows,
        "current_induction_success_count": current_success,
        "candidate_induction_success_count": candidate_success,
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
