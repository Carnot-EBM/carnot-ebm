"""Experiment 5598: a properly-powered follow-on to exp5596/exp5597's contradictory single-draw
generator-size comparisons.

exp5596 (dense Qwen3.6-27B-MTP candidate, 2 games, 1 draw each) found the candidate BEAT the
current frozen Qwen3.5-9B-MTP generator. exp5597 (MoE Qwen3.6-35B-A3B-MTP candidate, same 2
games, 1 draw each) found the candidate LOST to the current generator. Both spec entries flagged
the same honest limitation: n=2 games, 1 draw per arm, is far below the CLAUDE.md sample-size
floor for any percentage-point claim, and the CURRENT arm's own baseline score differed between
the two runs on the identical (model, game, budget) configuration -- real LLM sampling variance,
not a bug. This experiment resolves that ambiguity by testing all THREE arms (current, both
candidates) together, on a WIDER roster, with MULTIPLE independent repeats per (arm, game) cell.

DESIGN. Roster widened from 2 to 4 games (m0r0, sk48, cd82, sp80 -- the first two already used in
exp5596/5597, the latter two reused from exp5591's blob-topology roster and independently
verified this session via a direct E3AgentPolicy probe before being added here, since they had
only previously been exercised with raw env.step(), not the induction pipeline). N_SEEDS=3
independent repeats per (arm, game) cell -- NOT a controlled RNG seed (LocalGGUFProposer's
completion calls don't expose a `seed` parameter and this experiment does not add one, to avoid
touching shared production code for a one-off measurement); "seed" here means an independent
draw under the model's own real sampling temperature, which is exactly the axis of variance that
made exp5596 and exp5597 disagree, so characterizing it directly is the right fix. This yields
4 games x 3 repeats = 12 independent draws per arm (36 total per arm-pair when comparing two),
materially more powered than the prior n=2 single-draw comparisons, though still below the
CLAUDE.md N>=30 floor for a firm percentage-point claim -- this experiment reports descriptive
statistics (mean, std, per-game breakdown, paired win/loss/tie counts against current), not a
significance test, and is explicit about that limitation in its own verdict framing.

EFFICIENCY. The loop is batched BY ARM (start one arm's server once, run all its games x repeats,
then stop and move to the next arm) rather than interleaved (exp5596/5597's per-attempt
current/candidate alternation), avoiding N-1 unnecessary server restarts. Each arm's server is
still explicitly stopped + `_wait_for_port_down`-polled before the next arm starts (exp5596's
GPU-pinning fix, reused), since all three arms share GPU 1.

MTP FEASIBILITY. Both candidates' self-draft MTP feasibility is re-checked here (not assumed from
exp5596/5597's findings) via the same two-step check those experiments built
(`_declares_mtp_metadata` + `_mtp_self_draft_fits_vram`, generalized to take a repo_substr
argument instead of a single hardcoded candidate).

Per the task's own guardrail (mirroring task 7's frozen-stack discipline): this is an OFFLINE DEV
MEASUREMENT, not a live-stack change. Do NOT flip the frozen live generator based on this
experiment alone; report the delta and require an explicit operator decision.

Spec refs: REQ-ARC-WMTE-5598, SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON,
SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE.
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

EXPERIMENT_ID = "experiment_5598_generator_size_multiseed_ab"
RESULT_RELATIVE_PATH = "results/experiment_5598_generator_size_multiseed_ab.json"
SCHEMA = "carnot.exp5598.generator_size_multiseed_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5598
DEFAULT_ROSTER = ("m0r0", "sk48", "cd82", "sp80")
DEFAULT_N_SEEDS = 3
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
SHARED_MAX_TOKENS = 2560

ARMS: tuple[JsonDict, ...] = (
    {
        "name": "current",
        "repo_substr": "Qwen3.5-9B-MTP",
        "port": 8941,
        "always_mtp": True,
        "n_ctx": None,
    },
    {
        "name": "candidate_27b",
        "repo_substr": "Qwen3.6-27B-MTP",
        "port": 8940,
        "always_mtp": False,
        "n_ctx": None,
    },
    {
        "name": "candidate_35b_moe",
        "repo_substr": "Qwen3.6-35B-A3B-MTP",
        "port": 8942,
        "always_mtp": False,
        # Reduced from the class default (16384): a first attempt at this arm crashed mid-run
        # (the server process died, then a coincident GPU-1 hardware fault took the card off the
        # PCI bus entirely -- root cause not fully isolated between the two). Whatever the exact
        # trigger, this arm's non-MTP single-load footprint was already tight (21.9GB used of
        # 24GB, ~2.2GB free -- confirmed by exp5597's own manual sanity check), so cutting the KV
        # cache allocation via a smaller n_ctx gives real headroom for this retry rather than
        # re-running at the same razor-thin margin. 10240 is well above what m0r0/sk48/cd82/sp80
        # have needed at n_ctx=16384 in every prior induction call this session (none showed
        # context-overflow errors), so this should not introduce a new failure mode.
        "n_ctx": 10240,
    },
)

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "current frozen live-submission generator (arm=current)",
    },
    {
        "name": "Qwen3.6-27B-MTP",
        "hf_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "role": "dense candidate, exp5596's original candidate (arm=candidate_27b)",
    },
    {
        "name": "Qwen3.6-35B-A3B-MTP",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "role": "MoE candidate, exp5597's candidate (arm=candidate_35b_moe)",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "roster",
    "n_seeds",
    "arm_mtp_used",
    "per_draw_results",
    "per_arm_summary",
    "paired_vs_current",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a ranked, descriptive summary across 3 arms, NOT a "
        "significance claim -- n=12 draws/arm is more powered than exp5596/5597's n=2 but still "
        "below the CLAUDE.md N>=30 floor for a firm percentage-point claim"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- all arms invoke a real local GGUF proposer, not mocked"
    },
    "per_draw_results": {
        "principle": "every individual (arm, game, repeat) draw is recorded, not just aggregates "
        "-- exp5596/5597's contradiction came from single draws, so preserving the full draw "
        "list is what lets a reader assess variance directly rather than trusting a summary alone"
    },
    "paired_vs_current": {
        "principle": "per-(game, repeat) win/loss/tie counts against the current arm, the "
        "higher-power paired comparison (controls for per-game/per-draw difficulty) vs comparing "
        "unpaired means across arms"
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
    for arm in ARMS:
        key = f"gguf_cached_{arm['name']}"
        checks[key] = (
            any(str(arm["repo_substr"]).lower() in p.name.lower() for p in hub.glob("models--*"))
            if hub.exists()
            else False
        )
    checks["llama_server_binary_present"] = bool(
        list((Path.home() / ".cache").glob("llama.cpp*/build/bin/llama-server"))
    )
    checks["gpu1_free_vram_sufficient"] = _gpu1_free_mb() >= 20000
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


def _declares_mtp_metadata(repo_substr: str) -> bool:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    path = _resolve_gguf(repo_substr)
    if not path:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(1 << 20)
        return b"nextn_predict_layers" in header or b"n_predict_layers" in header
    except Exception:
        return False


def _mtp_self_draft_fits_vram(repo_substr: str) -> tuple[bool, str]:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    path = _resolve_gguf(repo_substr)
    if not path:
        return False, "GGUF not resolvable"
    try:
        file_mb = Path(path).stat().st_size / (1024 * 1024)
    except Exception:
        return False, "could not stat GGUF file size"
    self_draft_estimate_mb = 2.0 * file_mb
    free_mb = _gpu1_free_mb()
    if free_mb < 0:
        return False, "GPU 1 free VRAM unreadable"
    fits = free_mb >= (self_draft_estimate_mb + 2000)
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


def _make_proposer(
    arm_name: str, repo_substr: str, port: int, mtp_used: bool, n_ctx: int | None
) -> Any:
    """Construct ONE proposer per arm (called once, before that arm's game/repeat loops), so
    every draw within the arm reuses the SAME already-loaded server
    (`LocalGGUFProposer._ensure_server`'s existing reuse-if-healthy behavior) and the caller
    holds a real handle to call `.stop()` on when the arm's draws are done -- fixes an earlier
    draft that constructed a throwaway proposer per draw with no way to actually stop the
    server between arms, silently leaving every arm's server running and starving the next
    arm's free-VRAM guard (exp5596's GPU-pinning bug, reintroduced by that draft and caught in
    review before ever being run). `n_ctx=None` uses LocalGGUFProposer's own default (16384);
    a smaller explicit value trades context budget for VRAM headroom (see ARMS's
    candidate_35b_moe entry for why this arm needs it)."""

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    kwargs: JsonDict = {
        "repo_substr": repo_substr,
        "port": port,
        "mtp": mtp_used,
        "kv_quant": "q8_0",
        "max_tokens": SHARED_MAX_TOKENS,
    }
    if n_ctx is not None:
        kwargs["n_ctx"] = n_ctx
    if arm_name == "current":
        kwargs["no_think_prefix"] = "/no_think\n"
    return LocalGGUFProposer(**kwargs)


def _run_one_draw(
    game: str,
    *,
    arm_name: str,
    proposer: Any,
    repeat: int,
    explore_budget: int,
    total_budget: int,
) -> JsonDict:
    import os

    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = "1"

    policy = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
    lb.run_game(game, policy, budget=total_budget)
    active_transitions = list(policy.transitions)[:10]

    row: JsonDict = {"arm": arm_name, "game": game, "repeat": repeat}
    if not active_transitions:
        row.update({"transition_count": 0, "induction_ok": False})
        return row

    induce_started = time.time()
    ok, detail = proposer.induce(policy.short, active_transitions, policy.cell)
    row.update(
        {
            "transition_count": len(active_transitions),
            "induction_ok": bool(ok),
            "induce_duration_s": round(time.time() - induce_started, 3),
        }
    )
    if not ok:
        row["induction_failure_detail"] = str(detail)[:300]
        return row

    engine, _is_level_complete = e3.load_engine(policy.short)
    verify_result = WorldModelVerifier(active_transitions).score(engine)
    row["heldout_accuracy"] = verify_result.accuracy
    row["cell_recall"] = verify_result.cell_recall
    return row


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    n_seeds: int = DEFAULT_N_SEEDS,
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
            "roster": list(roster),
            "n_seeds": int(n_seeds),
            "arm_mtp_used": {},
            "per_draw_results": [],
            "per_arm_summary": {},
            "paired_vs_current": {},
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

    arm_mtp_used: dict[str, bool] = {}
    rows: list[JsonDict] = []
    for arm in ARMS:
        arm_name = str(arm["name"])
        repo_substr = str(arm["repo_substr"])
        port = int(arm["port"])
        if arm["always_mtp"]:
            mtp_used = True
        else:
            declares = _declares_mtp_metadata(repo_substr)
            fits, _detail = _mtp_self_draft_fits_vram(repo_substr)
            mtp_used = declares and fits
        arm_mtp_used[arm_name] = mtp_used

        # ONE proposer for the whole arm: every draw reuses the same already-loaded server
        # (LocalGGUFProposer._ensure_server's reuse-if-healthy path), and this object is the
        # real handle used to stop the server once, after all this arm's draws are done.
        arm_n_ctx = arm.get("n_ctx")
        proposer = _make_proposer(arm_name, repo_substr, port, mtp_used, arm_n_ctx)
        gpu1_lost_mid_run = False
        try:
            for game in roster:
                if gpu1_lost_mid_run:
                    break
                for repeat in range(n_seeds):
                    # GPU-1-HEALTH GUARD (found necessary: a real run's GPU 1 fell off the PCI
                    # bus mid-arm -- nvidia-smi could no longer query it at all -- and the
                    # existing generator fallback logic silently switched to the slow iGPU rather
                    # than erroring, which would have contaminated this arm's draws with a
                    # different, inconsistent hardware tier partway through. Check BEFORE each
                    # draw and fail closed (stop collecting, mark the run honestly) rather than
                    # silently continuing on degraded hardware.
                    if _gpu1_free_mb() < 0:
                        gpu1_lost_mid_run = True
                        rows.append(
                            {
                                "arm": arm_name,
                                "game": game,
                                "repeat": repeat,
                                "error": "gpu1_unreachable_mid_run_aborting_remaining_draws",
                            }
                        )
                        break
                    try:
                        rows.append(
                            _run_one_draw(
                                game,
                                arm_name=arm_name,
                                proposer=proposer,
                                repeat=repeat,
                                explore_budget=explore_budget,
                                total_budget=total_budget,
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "arm": arm_name,
                                "game": game,
                                "repeat": repeat,
                                "error": repr(exc)[:200],
                            }
                        )
        finally:
            # STOP this arm's server before the NEXT arm starts -- all arms share GPU 1; a
            # still-resident server would starve the next arm's launch (exp5596's fix, reused).
            proposer.stop()
            _wait_for_port_down(port)
        if gpu1_lost_mid_run:
            break

    per_arm_summary: JsonDict = {}
    for arm in ARMS:
        arm_name = str(arm["name"])
        arm_rows = [r for r in rows if r.get("arm") == arm_name]
        successes = [r for r in arm_rows if r.get("induction_ok")]
        accuracies = [float(r["heldout_accuracy"]) for r in successes if "heldout_accuracy" in r]
        per_game: JsonDict = {}
        for game in roster:
            game_accuracies = [
                float(r["heldout_accuracy"])
                for r in successes
                if r.get("game") == game and "heldout_accuracy" in r
            ]
            per_game[game] = {
                "n_draws": len(game_accuracies),
                "mean_accuracy": round(_mean(game_accuracies), 4),
                "std_accuracy": round(_std(game_accuracies), 4),
            }
        per_arm_summary[arm_name] = {
            "n_attempted": len(arm_rows),
            "n_induction_ok": len(successes),
            "mean_accuracy": round(_mean(accuracies), 4),
            "std_accuracy": round(_std(accuracies), 4),
            "per_game": per_game,
        }

    paired_vs_current: JsonDict = {}
    current_by_key = {
        (r["game"], r["repeat"]): r
        for r in rows
        if r.get("arm") == "current" and "heldout_accuracy" in r
    }
    for arm in ARMS:
        arm_name = str(arm["name"])
        if arm_name == "current":
            continue
        wins = losses = ties = 0
        for r in rows:
            if r.get("arm") != arm_name or "heldout_accuracy" not in r:
                continue
            cur = current_by_key.get((r["game"], r["repeat"]))
            if cur is None:
                continue
            if r["heldout_accuracy"] > cur["heldout_accuracy"]:
                wins += 1
            elif r["heldout_accuracy"] < cur["heldout_accuracy"]:
                losses += 1
            else:
                ties += 1
        paired_vs_current[arm_name] = {"wins": wins, "losses": losses, "ties": ties}

    ranked = sorted(
        (arm["name"] for arm in ARMS),
        key=lambda name: per_arm_summary[name]["mean_accuracy"],
        reverse=True,
    )
    total_draws_ok = sum(s["n_induction_ok"] for s in per_arm_summary.values())
    gpu1_lost_any_arm = any(
        r.get("error") == "gpu1_unreachable_mid_run_aborting_remaining_draws" for r in rows
    )
    if gpu1_lost_any_arm:
        # Honest partial result: GPU 1 became unreachable mid-run and the run was aborted rather
        # than silently continuing on inconsistent hardware. Whatever arms completed BEFORE the
        # loss are still real, valid, same-hardware-tier data (per_arm_summary/paired_vs_current
        # reflect them correctly) -- only the verdict prefix signals the run is incomplete, so a
        # reader does not mistake a partial ranking for the full 3-arm comparison.
        verdict = (
            f"complete: generator_size_multiseed_ab_blocked_gpu1_lost_mid_run_partial_ranked_"
            f"{'_gt_'.join(ranked)}"
        )
    elif total_draws_ok == 0:
        verdict = "complete: generator_size_multiseed_ab_no_successful_draws_inconclusive"
    else:
        verdict = f"complete: generator_size_multiseed_ab_ranked_{'_gt_'.join(ranked)}"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "roster": list(roster),
        "n_seeds": int(n_seeds),
        "arm_mtp_used": arm_mtp_used,
        "per_draw_results": rows,
        "per_arm_summary": per_arm_summary,
        "paired_vs_current": paired_vs_current,
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
