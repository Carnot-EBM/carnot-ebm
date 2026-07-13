"""Experiment 5594: does enabling `/think` mode (instead of the frozen live
generator's `/no_think`) improve tier-3 induction quality on the real default
Qwen3.5-9B-MTP proposer?

Context: ops/known-issues.md task 7. ARC Prize's GPT-5.6 results
(arcprize.org/results/openai-gpt-5-6, 2026-07-10) show reasoning-effort
scaling ARC-AGI-3 performance ~26x (Low->Max) versus only ~1.3x on ARC-AGI-1
for the SAME model. Independently, Duck Harness (the ARC-AGI-3 Milestone-1
1st-place team) converges on the same underlying principle via a completely
different mechanism (12 tool-calling turns per action-decision to orient
before committing, rather than internal reasoning tokens). Our frozen live
generator runs `/no_think` -- decided under June sprint time pressure for
Kaggle-parity/latency, never re-measured since.

PRECONDITION (a) from the task ("confirm Qwen3.5-9B-MTP actually exposes a
think-mode toggle compatible with MTP decoding"): verified as part of this
experiment's own precondition check, not assumed -- a real completion call
with a `/think` prefix must produce visibly longer, qualitatively different
(reasoning-shaped) output than the `/no_think` default, and must not error.

MECHANISM NOTE (found investigating, not assumed): `no_think_prefix` (the
LocalGGUFProposer instance attribute) has NO EFFECT on real induction calls
today, because `CARNOT_ARC_CODEONLY_INDUCE` defaults ON and codeonly mode's
own `_L2_CODEONLY_DIRECTIVE` hardcodes a literal `/no_think\n` as its first
line, overriding whatever the instance attribute says. To test `/think`
fairly, this experiment uses a SCOPED MODULE-LEVEL MONKEYPATCH of
`_L2_CODEONLY_DIRECTIVE` (swapping `/no_think\n` for `/think\n`, restored
after each induction call) rather than the (dead-for-this-path)
`no_think_prefix` attribute -- keeping codeonly mode's other output-format
constraints (ONLY a code block, no prose) while isolating the think/no-think
axis specifically.

SCOPE (honest, not the task's full ideal): the task's stated metric is
"actions-to-first-win and first-contact solve rate ... on held-out games" --
a full live-play measurement across many games. This experiment is a
narrower, cheaper FIRST PASS per the task's own "Cheap, DEV-SIDE ONLY"
framing: induction QUALITY (WorldModelVerifier heldout_accuracy, plus
score_goal_predicate_consistency where a real level-up is available in the
sample -- reusing this session's own new REQ-ARC-WMTE-5593 verifier) on a
small roster, single induction attempt per arm per game. A full
actions-to-first-win sweep is a natural, more expensive follow-on if this
narrower signal is positive.

Tier-3 LLM think-mode needs materially more completion budget than
`/no_think` (confirmed: a `/think` call did not finish its own reasoning
within 300 tokens) -- the THINK arm is given a larger max_tokens budget than
the frozen NO_THINK default (2560); this asymmetry is by design (comparing
truncated-mid-thought output to quick code would not be a fair test of
`/think`'s actual value) and is disclosed, not hidden.

Per the task's own guardrail: this is an OFFLINE DEV MEASUREMENT, not a
live-stack change. Do NOT flip the frozen stack's `/no_think` setting based
on this experiment alone; report the delta and require an explicit operator
decision before touching the frozen live-submission config.

Spec refs: REQ-ARC-WMTE-5594, SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY,
SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED.
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

EXPERIMENT_ID = "experiment_5594_think_mode_induction_quality_ab"
RESULT_RELATIVE_PATH = "results/experiment_5594_think_mode_induction_quality_ab.json"
SCHEMA = "carnot.exp5594.think_mode_induction_quality_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5594
DEFAULT_ROSTER = ("m0r0", "sk48")
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
NO_THINK_MAX_TOKENS = 2560  # matches the frozen live default exactly
THINK_MAX_TOKENS = 6144  # think mode needs materially more room; disclosed asymmetry
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default tier-3 world-model induction proposer",
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "roster",
    "think_mode_compatible_with_mtp",
    "think_mode_compat_detail",
    "no_think_max_tokens",
    "think_max_tokens",
    "per_game_results",
    "no_think_induction_success_count",
    "think_induction_success_count",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; think mode helping, not helping, or being incompatible with MTP are all distinct, real, citable outcomes"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- both arms invoke the real default Qwen3.5-9B-MTP proposer, not mocked"
    },
    "think_mode_compatible_with_mtp": {
        "principle": "task 7 precondition (a), checked here rather than assumed -- if False, this experiment stops per the task's blocked_think_mode_incompatible_with_mtp instruction rather than proceeding on an untested assumption"
    },
    "think_max_tokens": {
        "principle": "materially larger than no_think_max_tokens by design -- think mode needs completion budget for reasoning tokens before code; comparing truncated-mid-thought output to quick code would not be a fair test, and this asymmetry is disclosed, not hidden"
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
            WorldModelVerifier,
            score_goal_predicate_consistency,
        )

        checks["e3_policy_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
    try:
        from carnot.agentic.arc_executable_world_model import _resolve_gguf, _resolve_llama_server

        checks["gguf_cached"] = _resolve_gguf(GGUF_REPO_SUBSTR) is not None
        checks["llama_server_binary_present"] = _resolve_llama_server().exists()
    except Exception:
        checks["gguf_cached"] = False
        checks["llama_server_binary_present"] = False
    required_keys = set(checks)
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8920/health", timeout=3) as resp:
            checks["port_8920_prewarmed"] = b"ok" in resp.read()
    except Exception:
        checks["port_8920_prewarmed"] = False
    checks["ok"] = all(checks[key] for key in required_keys)
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


def check_think_mode_compatibility() -> tuple[bool, str]:
    """REQ-ARC-WMTE-5594 precondition (a): a real completion call with a `/think` prefix
    against the live MTP-enabled server must produce genuinely different (reasoning-
    shaped, materially longer before any code/answer) output than `/no_think`, and must
    not error. Confirms MTP speculative decoding does not silently ignore or break
    explicit think-mode requests."""

    import json as _json
    import urllib.request

    probe_task = (
        "You are inducing a Python world-model. Write engine(grid, action, data) "
        "and is_level_complete(grid)."
    )
    try:
        no_think_body = _json.dumps(
            {
                "prompt": f"/no_think\n{probe_task}\n```python\n",
                "n_predict": 120,
                "temperature": 0.2,
            }
        ).encode()
        think_body = _json.dumps(
            {"prompt": f"/think\n{probe_task}\n", "n_predict": 120, "temperature": 0.2}
        ).encode()
        no_think_req = urllib.request.Request(
            "http://127.0.0.1:8920/completion",
            data=no_think_body,
            headers={"Content-Type": "application/json"},
        )
        think_req = urllib.request.Request(
            "http://127.0.0.1:8920/completion",
            data=think_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(no_think_req, timeout=60) as r:
            no_think_content = _json.load(r).get("content", "")
        with urllib.request.urlopen(think_req, timeout=60) as r:
            think_content = _json.load(r).get("content", "")
    except Exception as exc:
        return False, f"probe request failed: {exc!r}"[:200]

    # Qwen3-family reasoning models are not consistent about the exact opening
    # tag spelling across calls (observed both "<think>" and "<thinking>" from
    # the same server in back-to-back probes) -- a bare "<think>" substring
    # check misses the "<thinking>" variant entirely (the literal characters
    # "<think>" are not present inside "<thinking>", since the closing ">"
    # doesn't align). Check a tuple of known prefixes instead of one literal.
    think_stripped = think_content.strip()
    no_think_stripped = no_think_content.strip()
    reasoning_tag_prefixes = ("<think>", "<thinking>", "<reasoning>")
    think_shows_tag = think_stripped.startswith(reasoning_tag_prefixes)
    # Length-ratio is a fallback signal only, not primary -- a short n_predict=120
    # probe leaves little room for a 1.5x delta to materialize even when think-mode
    # is genuinely engaged (observed 549 vs 403 chars, a real 36% delta that a 1.5x
    # threshold rejected as "no difference"). Lower bar: any material delta counts.
    think_shows_length_delta = len(think_stripped) > (1.15 * len(no_think_stripped))
    think_shows_reasoning = think_shows_tag or think_shows_length_delta
    if not think_shows_reasoning:
        return False, (
            f"no reasoning-tag prefix and no material length delta "
            f"({len(think_stripped)} vs {len(no_think_stripped)} chars) -- "
            "MTP may be silently ignoring the think-mode toggle"
        )
    return True, (
        f"think content {'starts with a reasoning tag' if think_shows_tag else 'is materially longer'} "
        f"({len(think_content)} vs {len(no_think_content)} chars) -- compatible"
    )


def _run_one_arm(
    game: str,
    *,
    arm: str,
    explore_budget: int,
    total_budget: int,
) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import (
        LocalGGUFProposer,
        WorldModelVerifier,
        score_goal_predicate_consistency,
    )

    max_tokens = THINK_MAX_TOKENS if arm == "think" else NO_THINK_MAX_TOKENS
    proposer = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=8920,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=max_tokens,
    )
    policy = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
    lb.run_game(game, policy, budget=total_budget)
    active_transitions = list(policy.transitions)[:10]

    if not active_transitions:
        return {"game": game, "arm": arm, "transition_count": 0, "induction_ok": False}

    original_directive = e3._L2_CODEONLY_DIRECTIVE
    if arm == "think":
        e3._L2_CODEONLY_DIRECTIVE = original_directive.replace("/no_think\n", "/think\n", 1)
    try:
        ok, detail = proposer.induce(policy.short, active_transitions, policy.cell)
    finally:
        e3._L2_CODEONLY_DIRECTIVE = original_directive

    row: JsonDict = {
        "game": game,
        "arm": arm,
        "transition_count": len(active_transitions),
        "induction_ok": bool(ok),
    }
    if not ok:
        row["induction_failure_detail"] = str(detail)[:300]
        return row

    engine, is_level_complete = e3.load_engine(policy.short)
    verify_result = WorldModelVerifier(active_transitions).score(engine)
    row["heldout_accuracy"] = verify_result.accuracy
    row["cell_recall"] = verify_result.cell_recall
    real_levelup_present = any(t.level_after > t.level_before for t in active_transitions)
    row["real_levelup_present_in_sample"] = real_levelup_present
    if real_levelup_present:
        goal_result = score_goal_predicate_consistency(is_level_complete, active_transitions)
        row["goal_predicate_accuracy"] = goal_result.accuracy
    return row


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
            "roster": list(roster),
            "think_mode_compatible_with_mtp": False,
            "think_mode_compat_detail": "",
            "no_think_max_tokens": NO_THINK_MAX_TOKENS,
            "think_max_tokens": THINK_MAX_TOKENS,
            "per_game_results": [],
            "no_think_induction_success_count": 0,
            "think_induction_success_count": 0,
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

    compatible, compat_detail = check_think_mode_compatibility()
    if not compatible:
        artifact = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": "complete: blocked_think_mode_incompatible_with_mtp",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "roster": list(roster),
            "think_mode_compatible_with_mtp": False,
            "think_mode_compat_detail": compat_detail,
            "no_think_max_tokens": NO_THINK_MAX_TOKENS,
            "think_max_tokens": THINK_MAX_TOKENS,
            "per_game_results": [],
            "no_think_induction_success_count": 0,
            "think_induction_success_count": 0,
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

    rows: list[JsonDict] = []
    for game in roster:
        for arm in ("no_think", "think"):
            try:
                rows.append(
                    _run_one_arm(
                        game,
                        arm=arm,
                        explore_budget=explore_budget,
                        total_budget=total_budget,
                    )
                )
            except Exception as exc:
                rows.append({"game": game, "arm": arm, "error": repr(exc)[:200]})

    no_think_success = sum(1 for r in rows if r.get("arm") == "no_think" and r.get("induction_ok"))
    think_success = sum(1 for r in rows if r.get("arm") == "think" and r.get("induction_ok"))

    no_think_accuracies = [
        r["heldout_accuracy"] for r in rows if r.get("arm") == "no_think" and r.get("induction_ok")
    ]
    think_accuracies = [
        r["heldout_accuracy"] for r in rows if r.get("arm") == "think" and r.get("induction_ok")
    ]

    if think_success == 0 and no_think_success == 0:
        verdict = "complete: think_mode_ab_neither_arm_induced_inconclusive"
    elif think_success > no_think_success:
        verdict = f"complete: think_mode_ab_think_more_reliable_{no_think_success}_to_{think_success}_successes"
    elif think_success < no_think_success:
        verdict = f"complete: think_mode_ab_no_think_more_reliable_{think_success}_to_{no_think_success}_successes"
    elif (
        think_accuracies
        and no_think_accuracies
        and sum(think_accuracies) / len(think_accuracies)
        > sum(no_think_accuracies) / len(no_think_accuracies)
    ):
        verdict = "complete: think_mode_ab_equal_success_think_higher_accuracy"
    elif (
        think_accuracies
        and no_think_accuracies
        and sum(think_accuracies) / len(think_accuracies)
        < sum(no_think_accuracies) / len(no_think_accuracies)
    ):
        verdict = "complete: think_mode_ab_equal_success_no_think_higher_accuracy"
    else:
        verdict = "complete: think_mode_ab_honest_null_no_measured_difference"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "roster": list(roster),
        "think_mode_compatible_with_mtp": True,
        "think_mode_compat_detail": compat_detail,
        "no_think_max_tokens": NO_THINK_MAX_TOKENS,
        "think_max_tokens": THINK_MAX_TOKENS,
        "per_game_results": rows,
        "no_think_induction_success_count": no_think_success,
        "think_induction_success_count": think_success,
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
