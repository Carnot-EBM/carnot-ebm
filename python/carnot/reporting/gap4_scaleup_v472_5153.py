"""Exp 5153 GAP-4 scale-up protocol ledger.

This module does not pretend that the expensive 400-task GAP-4 run happened.
It turns GAP-4's own forward protocol into a machine-checkable result artifact:
the prior +4/-0 positive is preserved as context, but the status recommendation
stays `still_open` until every protocol step is actually satisfied.

Spec refs: REQ-VERIFY-5153, SCENARIO-VERIFY-5153,
SCENARIO-VERIFY-5153-SUCCESS-GATE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


OUTPUT_REL = Path("results/experiment_5153_gap4_scaleup_v472.json")
SCHEMA = "carnot.gap4_scaleup_v472_5153.v1"
EXPERIMENT = "experiment_5153_gap4_scaleup_v472"
SOLVE_PROVENANCE = "development_proxy"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
STATUS_RECOMMENDATIONS = {"filled", "still_open", "retired"}
SPEC_REFS = [
    "REQ-VERIFY-5153",
    "SCENARIO-VERIFY-5153",
    "SCENARIO-VERIFY-5153-SUCCESS-GATE",
]

CANONICAL_PROTOCOL_STEP_IDS = (
    "sandboxed_400_task_reconfirmation",
    "transcripts_archived",
    "genuinely_heldout_tasks",
    "codex_first_arm",
    "statistical_tests",
    "hardened_exec_sandbox",
    "local_open_weight_generator_arm",
)

FIELD_PRINCIPLES = {
    "protocol_steps_completed": (
        "GAP-4's own protocol is the acceptance bar; partial completion should be reported "
        "as partial, not rounded up to filled."
    ),
    "n_400_task_result": (
        "The actual scaled solve-rate/precision number, not just a pass/fail on the protocol steps."
    ),
    "gap4_status_recommendation": (
        "Feeds directly into whether ops/verifier_gaps.md's GAP-4 status line gets updated."
    ),
    "solve_provenance": "Offline scoring against held-out corpora, not a live hidden-game solve.",
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    "protocol_steps_completed",
    "n_400_task_result",
    "gap4_status_recommendation",
    "solve_provenance",
    "protocol_acceptance_passed",
    "cluster_bootstrap_delta_ci95",
    "exact_test_discordant_wins",
    "exact_test_discordant_losses",
    "exact_test_p_value",
    "exact_test_passes_min6_rule",
    "prior_positive_context",
    "field_principles",
    "spec_refs",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class ProtocolStep:
    """One GAP-4 forward-protocol checkpoint with evidence."""

    step_id: str
    passed: bool
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {"step_id": self.step_id, "passed": self.passed, "evidence": self.evidence}


def exact_two_sided_sign_p_value(wins: int, losses: int) -> float:
    """Return the two-sided exact sign-test p-value for discordant wins/losses."""

    if wins < 0 or losses < 0:
        raise ValueError("discordant win/loss counts must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / float(2**n)
    return min(1.0, probability)


def exact_test_passes_min6_rule(wins: int, losses: int) -> bool:
    """GAP-4 zero-loss acceptance rule: at least 6 wins, no losses, p<0.05."""

    return wins >= 6 and losses == 0 and exact_two_sided_sign_p_value(wins, losses) < 0.05


def default_protocol_steps(root: Path) -> list[ProtocolStep]:
    sandbox_code = root / "scripts/experiments/arc3_gap4_rule_exec_verifier.py"
    sandbox_tests = root / "tests/python/test_arc3_gap4_rule_exec.py"
    hardened_sandbox_present = sandbox_code.exists() and sandbox_tests.exists()
    return [
        ProtocolStep(
            "sandboxed_400_task_reconfirmation",
            False,
            "No new sandboxed 400-task run on a host attested to have no ARC solutions on disk.",
        ),
        ProtocolStep(
            "transcripts_archived",
            False,
            "Prior GAP-4 transcripts exist, but no transcripts for the required 400-task scale-up.",
        ),
        ProtocolStep(
            "genuinely_heldout_tasks",
            False,
            "The ARC-2 transfer probe is reduced-exposure calibration; no ARC-AGI-2 eval scale-up, ConceptARC holdout, or post-cutoff run is archived here.",
        ),
        ProtocolStep(
            "codex_first_arm",
            False,
            "The 2026-06-09/10 calibration used Codex-first, but no Codex-first 400-task reconfirmation was run for Exp 5153.",
        ),
        ProtocolStep(
            "statistical_tests",
            False,
            "Prior +4/-0 evidence is below the two-sided min-6 exact-test rule and no 400-task cluster bootstrap exists.",
        ),
        ProtocolStep(
            "hardened_exec_sandbox",
            hardened_sandbox_present,
            "Existing GAP-4 sandbox code/tests block timeout, np.load, np.save, np.fromfile, and type( vectors."
            if hardened_sandbox_present
            else "The GAP-4 sandbox code/tests were not present under this root.",
        ),
        ProtocolStep(
            "local_open_weight_generator_arm",
            False,
            "Existing local-generator attempts did not establish the decentralization tier, and no Gemma-4/Qwen3.6 scale-up arm is archived for Exp 5153.",
        ),
    ]


def prior_positive_context(root: Path) -> dict[str, Any]:
    return {
        "gap_id": "GAP-4",
        "gap_source": "ops/verifier_gaps.md",
        "verifier_gaps_present": (root / "ops/verifier_gaps.md").exists(),
        "arc1_rule_exec_artifact_present": (root / "results/arc3_gap4_rule_exec_verifier.json").exists(),
        "arc1_induced_programs_present": (root / "results/arc3_gap4_induced_programs.json").exists(),
        "arc2_transfer_artifact_present": (
            root / "results/arc3_gap4_arc2_rule_exec_verifier.json"
        ).exists(),
        "arc1_vote_pass2": 0.4516,
        "arc1_gated_pass2": 0.5806,
        "arc1_headroom_recovered": 4,
        "arc1_vote_wins_lost": 0,
        "arc2_induction_unique_tasks": [0.93, 0.57],
        "arc2_precision_given_demo_perfect": [0.90, 0.47],
        "calibration_read": (
            "Real prior positive and ARC-2 calibration preserved as upstream context; "
            "not sufficient for GAP-4 filled status without the forward protocol."
        ),
    }


def _checksum(payload: dict[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _terminal_prefixed(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _coerce_metric(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("n_400_task_result must be a float or null")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError("n_400_task_result must be finite")
    return metric


def _all_protocol_steps_pass(steps: Sequence[ProtocolStep]) -> bool:
    return bool(steps) and all(step.passed for step in steps)


def _status_recommendation(
    *,
    steps: Sequence[ProtocolStep],
    n_400_task_result: float | None,
    exact_test_passed: bool,
) -> str:
    if _all_protocol_steps_pass(steps) and n_400_task_result is not None and exact_test_passed:
        return "filled"
    return "still_open"


def build_artifact(
    *,
    protocol_steps: Sequence[ProtocolStep],
    n_400_task_result: float | int | None,
    exact_test_discordant_wins: int,
    exact_test_discordant_losses: int,
    cluster_bootstrap_delta_ci95: list[float] | None,
    prior_positive_context: dict[str, Any],
) -> dict[str, Any]:
    steps = list(protocol_steps)
    metric = _coerce_metric(n_400_task_result)
    p_value = exact_two_sided_sign_p_value(
        int(exact_test_discordant_wins), int(exact_test_discordant_losses)
    )
    exact_passed = exact_test_passes_min6_rule(
        int(exact_test_discordant_wins), int(exact_test_discordant_losses)
    )
    status = _status_recommendation(
        steps=steps, n_400_task_result=metric, exact_test_passed=exact_passed
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "honest_verdict": (
            "success_gap4_scaleup_protocol_complete_filled_recommended"
            if status == "filled"
            else "complete: gap4_scaleup_protocol_partial_still_open"
        ),
        "protocol_steps_completed": [step.as_dict() for step in steps],
        "n_400_task_result": metric,
        "gap4_status_recommendation": status,
        "solve_provenance": SOLVE_PROVENANCE,
        "protocol_acceptance_passed": status == "filled",
        "cluster_bootstrap_delta_ci95": cluster_bootstrap_delta_ci95,
        "exact_test_discordant_wins": int(exact_test_discordant_wins),
        "exact_test_discordant_losses": int(exact_test_discordant_losses),
        "exact_test_p_value": p_value,
        "exact_test_passes_min6_rule": exact_passed,
        "prior_positive_context": prior_positive_context,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(payload: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        _require(field in payload, f"missing required field: {field}")
    _require(
        _terminal_prefixed(str(payload["honest_verdict"])),
        "honest_verdict must be terminal-prefixed",
    )
    _require(
        payload["gap4_status_recommendation"] in STATUS_RECOMMENDATIONS,
        "status recommendation is invalid",
    )
    _require(
        payload["solve_provenance"] == SOLVE_PROVENANCE,
        "solve_provenance must equal development_proxy",
    )
    _require(
        not isinstance(payload["n_400_task_result"], bool),
        "n_400_task_result must be float or null",
    )
    _require(
        payload["n_400_task_result"] is None
        or isinstance(payload["n_400_task_result"], int | float),
        "n_400_task_result must be float or null",
    )
    _require(
        isinstance(payload["exact_test_discordant_wins"], int)
        and payload["exact_test_discordant_wins"] >= 0,
        "discordant wins must be a non-negative int",
    )
    _require(
        isinstance(payload["exact_test_discordant_losses"], int)
        and payload["exact_test_discordant_losses"] >= 0,
        "discordant losses must be a non-negative int",
    )
    steps = payload["protocol_steps_completed"]
    _require(isinstance(steps, list), "protocol_steps_completed must be a list")
    step_ids = []
    for step in steps:
        _require(isinstance(step, dict), "step records must be objects")
        _require(
            set(step) == {"step_id", "passed", "evidence"},
            "step records must contain step_id, passed, and evidence",
        )
        _require(
            isinstance(step["step_id"], str) and step["step_id"] in CANONICAL_PROTOCOL_STEP_IDS,
            "step id is invalid",
        )
        _require(isinstance(step["passed"], bool), "step passed must be a bare bool")
        _require(
            isinstance(step["evidence"], str) and bool(step["evidence"]),
            "step evidence is required",
        )
        step_ids.append(step["step_id"])
    _require(
        set(step_ids) == set(CANONICAL_PROTOCOL_STEP_IDS),
        "protocol step set is incomplete",
    )
    _require(
        payload["field_principles"] == FIELD_PRINCIPLES,
        "field_principles do not match REQ-VERIFY-5153",
    )
    _require(
        payload["spec_refs"] == SPEC_REFS,
        "spec_refs do not match REQ-VERIFY-5153",
    )
    _require(
        payload["reproducibility_checksum"] == _checksum(payload),
        "reproducibility_checksum mismatch",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(root: Path | str | None = None) -> dict[str, Any]:
    root_path = Path.cwd() if root is None else Path(root)
    artifact = build_artifact(
        protocol_steps=default_protocol_steps(root_path),
        n_400_task_result=None,
        exact_test_discordant_wins=4,
        exact_test_discordant_losses=0,
        cluster_bootstrap_delta_ci95=None,
        prior_positive_context=prior_positive_context(root_path),
    )
    validate_artifact(artifact)
    write_json(root_path / OUTPUT_REL, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    run(args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
