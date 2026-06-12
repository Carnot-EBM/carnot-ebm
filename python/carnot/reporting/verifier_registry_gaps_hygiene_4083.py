"""Exp 4083 GAP-4 safety-gate replay and verifier-pivot hygiene.

Spec refs: REQ-VERIFY-4083, SCENARIO-VERIFY-4083.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4083_ARTIFACT_PATH = "results/experiment_4083_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_RULE_EXEC_PATH = base.ARC1_RULE_EXEC_PATH
PIVOT_EXP4079_PATH = "results/experiment_4079_verifier_reward_rft_eval_collect.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
PIVOT_GAP_ID = "GAP-TRAINING-VERIFIER-AS-REWARD-RFT-4079"
PIVOT_ROLE_ID = "verifier_as_reward_rft_4079"

EXPECTED_ARC1 = {
    "n": 31,
    "vote_pass2": 0.4516,
    "gated_pass2": 0.5806,
    "headroom_recovered": 4,
    "vote_wins_lost": 0,
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_arc1_reproduced",
    "pivot_outcome_recorded",
    "safety_gate_regression_passed",
    "registry_updated",
    "gaps_updated",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix summary of the cached ARC-1 replay and Exp 4079 pivot record.",
    "gap4_arc1_reproduced": (
        "Regression guard: the shipped GAP-4 demo-fit gate must reproduce vote 0.4516 "
        "to gated 0.5806 from cached candidates."
    ),
    "pivot_outcome_recorded": (
        "Bare bool; the .377 verifier-as-reward training-time outcome is represented in the gaps ledger."
    ),
    "safety_gate_regression_passed": (
        "Bare bool; true only when the ARC-1 gate reproduces the exact lift and loses zero vote wins."
    ),
    "registry_updated": "Bare bool; registry carries the Exp 4083 replay marker and training-time role.",
    "gaps_updated": "Bare bool; gaps ledger carries the Exp 4083 pivot block.",
    "inference_substrate": "Cached verifier candidates only; no Codex, GGUF, or live inference.",
}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay the shipped GAP-4 ARC-1 gate from its checked-in result artifact."""
    artifact = base._load_json(repo_root / ARC1_RULE_EXEC_PATH)
    return replay_gap4_arc1_fixture(artifact)


def replay_gap4_arc1_fixture(artifact: dict[str, Any]) -> dict[str, Any]:
    """Replay ARC-1 metrics from an already-loaded GAP-4 result-shaped object."""
    observed = {
        "n": int(artifact["n_tasks"]),
        "vote_pass2": _round4(artifact["rankers"]["TRM_VOTE"]["pass@2"]),
        "gated_pass2": _round4(artifact["rankers"]["GAP4_GATED"]["pass@2"]),
        "headroom_recovered": int(artifact["gates"]["headroom_recovered"]),
        "vote_wins_lost": int(artifact["gates"]["vote_wins_lost"]),
    }
    reproduced = observed == EXPECTED_ARC1
    return {
        "gap4_arc1_reproduced": reproduced,
        "safety_gate_regression_passed": reproduced and observed["vote_wins_lost"] == 0,
        "arc1_rule_exec": observed,
        "expected": {"arc1_rule_exec": deepcopy(EXPECTED_ARC1)},
    }


def classify_pivot_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4079 as a training-time verifier role outcome."""
    path = repo_root / PIVOT_EXP4079_PATH
    if not path.exists():
        return {
            "pivot_outcome_recorded": "verifier_as_reward_rft_pending",
            "status": "pending",
            "training_time_role": True,
            "artifact_path": PIVOT_EXP4079_PATH,
            "reason": "missing_exp4079_artifact",
            "honest_verdict": "",
        }
    return classify_pivot_outcome_fixture(base._load_json(path), PIVOT_EXP4079_PATH)


def classify_pivot_outcome_fixture(
    artifact: dict[str, Any],
    artifact_path: str = PIVOT_EXP4079_PATH,
) -> dict[str, Any]:
    """Classify an already-loaded Exp 4079-shaped artifact without overstating it."""
    verdict = str(artifact.get("honest_verdict", ""))
    status = str(artifact.get("status", "")).lower()
    if status == "blocked" or verdict.startswith("blocked_"):
        label = "verifier_as_reward_rft_blocked"
        normalized_status = "blocked"
    elif status == "complete" or verdict.startswith("complete:"):
        label = "verifier_as_reward_rft_complete"
        normalized_status = "complete"
    else:
        label = "verifier_as_reward_rft_accumulating"
        normalized_status = status or "accumulating"
    return {
        "pivot_outcome_recorded": label,
        "status": normalized_status,
        "training_time_role": True,
        "artifact_path": artifact_path,
        "honest_verdict": verdict,
        "gate_check_summary": str(artifact.get("gate_check_summary", "")),
        "blocked_at_layer": str(artifact.get("blocked_at_layer", "")),
        "arm_a_vs_b_delta": artifact.get("arm_a_vs_b_delta"),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    pivot_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, bool]]:
    """Return registry and gaps text with Exp 4083 outcomes represented idempotently."""
    updated_registry = deepcopy(registry)
    registry_changed = _ensure_gap4_eval(updated_registry, offline_replay)
    registry_changed = _ensure_training_time_role(updated_registry, pivot_outcome) or registry_changed

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4083-pivot",
        _pivot_gap_block(pivot_outcome),
    )
    gaps_ok = _gaps_contain_pivot(updated_gaps, pivot_outcome)
    pivot_recorded = gaps_ok and pivot_outcome["status"] != "pending"
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": gaps_ok,
            "pivot_outcome_recorded": pivot_recorded,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> bool:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - malformed ledgers are handled but not expected.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    eval_block = entry.setdefault("eval", {})
    arc1 = offline_replay.get("arc1_rule_exec", {})
    required = {
        "eval_exp_4083": EXP4083_ARTIFACT_PATH,
        "arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
        "arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
        "exp4083_gap4_arc1_reproduced": bool(offline_replay.get("gap4_arc1_reproduced")),
        "exp4083_arc1_safety_gate_regression_passed": bool(
            offline_replay.get("safety_gate_regression_passed")
        ),
    }
    changed = False
    for key, value in required.items():
        if eval_block.get(key) != value:
            eval_block[key] = value
            changed = True
    return changed


def _ensure_training_time_role(registry: dict[str, Any], pivot_outcome: dict[str, Any]) -> bool:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - _ensure_gap4_eval creates this first.
        return False
    old_roles = entry.get("training_time_roles", [])
    role = {
        "role_id": PIVOT_ROLE_ID,
        "experiment": PIVOT_EXP4079_PATH,
        "role": "training_time_reward_signal",
        "status": pivot_outcome["status"],
        "outcome": pivot_outcome["pivot_outcome_recorded"],
        "honest_verdict": pivot_outcome.get("honest_verdict", ""),
    }
    new_roles = [r for r in old_roles if r.get("role_id") != PIVOT_ROLE_ID] + [role]
    entry["training_time_roles"] = new_roles
    return old_roles != new_roles


def _pivot_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {PIVOT_GAP_ID}: training-time verifier role for verifier-as-reward RFT\n"
        f"- status: {outcome['pivot_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', PIVOT_EXP4079_PATH)}`; "
        f"status={outcome.get('status')}; honest_verdict={outcome.get('honest_verdict')}; "
        f"blocked_at_layer={outcome.get('blocked_at_layer', '')}; "
        f"gate_check_summary={outcome.get('gate_check_summary', '')}.\n"
        "- failure mode: the verifier-as-reward training-time role is not yet a decision-grade win; "
        "Exp 4079 must beat the cold base and the label ablation on held-out induction before promotion.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified training carries signal "
        "beyond codex-distillation and gold-SFT controls.\n"
        "- candidate design: unblock the train launch, rerun Exp 4079 on the held-out ladder, and keep "
        "the RFT-correct vs RFT-ablation contrast load-bearing.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if gap4 is None:  # pragma: no cover - defensive for malformed local ledgers.
        return False
    eval_block = gap4.get("eval", {})
    return (
        eval_block.get("eval_exp_4083") == EXP4083_ARTIFACT_PATH
        and "training_time_roles" in gap4
        and any(role.get("role_id") == PIVOT_ROLE_ID for role in gap4["training_time_roles"])
    )


def _gaps_contain_pivot(gaps_text: str, pivot_outcome: dict[str, Any]) -> bool:
    return PIVOT_GAP_ID in gaps_text and pivot_outcome["pivot_outcome_recorded"] in gaps_text


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    pivot_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: bool,
    pivot_outcome_recorded: bool,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4083 terminal JSON payload."""
    gap4_ok = bool(offline_replay.get("gap4_arc1_reproduced"))
    safety_ok = bool(offline_replay.get("safety_gate_regression_passed"))
    prefix = "complete:" if gap4_ok and pivot_outcome_recorded and safety_ok else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4083_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4083_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}gap4_arc1_reproduced_{gap4_ok}_"
            f"safety_gate_regression_{safety_ok}_pivot_{pivot_outcome['pivot_outcome_recorded']}"
        ),
        "gap4_arc1_reproduced": gap4_ok,
        "pivot_outcome_recorded": bool(pivot_outcome_recorded),
        "safety_gate_regression_passed": safety_ok,
        "registry_updated": bool(registry_updated),
        "gaps_updated": bool(gaps_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 3),
        "offline_replay": offline_replay,
        "pivot_outcome": pivot_outcome,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [ARC1_RULE_EXEC_PATH, PIVOT_EXP4079_PATH],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the fields the conductor expects before writing the result file."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    for field in (
        "gap4_arc1_reproduced",
        "pivot_outcome_recorded",
        "safety_gate_regression_passed",
        "registry_updated",
        "gaps_updated",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")  # pragma: no cover


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4083 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    pivot_outcome = classify_pivot_outcome(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        pivot_outcome,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        pivot_outcome=pivot_outcome,
        registry_updated=ledger_summary["registry_updated"],
        gaps_updated=ledger_summary["gaps_updated"],
        pivot_outcome_recorded=ledger_summary["pivot_outcome_recorded"],
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4083_ARTIFACT_PATH, artifact)
    return artifact


def _round4(value: float) -> float:
    return round(float(value), 4)


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4083_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
