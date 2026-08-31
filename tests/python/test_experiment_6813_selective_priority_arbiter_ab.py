"""Tests for the deterministic Exp6813 selective priority arbiter.

Spec refs: REQ-CONSTRAINT-6813 and SCENARIO-CONSTRAINT-6813-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_6813_selective_priority_arbiter_ab as exp


def _candidate(
    index: int,
    *,
    hard: int = 0,
    binding: list[int] | None = None,
    soft: int = 0,
    legal: bool = True,
    parse_state: str = "complete",
    first_conflict: str | None = None,
) -> dict:
    return {
        "candidate_index": index,
        "candidate_id": f"candidate_{index}",
        "candidate": {
            "action": {"data": {"index": index}, "kind": f"ACTION_{index}"},
            "authority_chain": ["authority"],
            "candidate_id": f"candidate_{index}",
            "soft_progress": soft,
        },
        "parse_state": parse_state,
        "parse_failure": None if parse_state == "complete" else "json_decode_error",
        "hard_violation_count": hard,
        "binding_violation_vector": binding or [],
        "binding_obligation_ids": [f"binding-{item}" for item in range(len(binding or []))],
        "soft_score": soft,
        "legal_support": legal,
        "first_conflict": first_conflict,
    }


def _select(candidates: list[dict], arm: str = exp.SELECTIVE_ARM, **kwargs: object) -> dict:
    return exp.select_arm(
        candidates,
        arm=arm,
        fallback_action={"data": None, "kind": "NOOP"},
        soft_score_clip=100,
        **kwargs,
    )


def test_scenario_constraint_6813_lexicographic_dominance() -> None:
    """SCENARIO-CONSTRAINT-6813-LEXICOGRAPHIC fixes priority order."""

    decision = _select(
        [
            _candidate(0, hard=1, soft=10**30, legal=False, first_conflict="hard-0"),
            _candidate(1, binding=[0, 1], soft=-100),
        ]
    )
    assert decision["selected_candidate_id"] == "candidate_1"
    assert decision["certificate"]["first_higher_priority_conflict"] == "hard-0"

    authority_order = _select(
        [
            _candidate(0, binding=[1, 0], soft=10**30),
            _candidate(1, binding=[0, 1], soft=-(10**30)),
        ]
    )
    assert authority_order["selected_candidate_id"] == "candidate_1"

    soft_order = _select([_candidate(0, binding=[1], soft=3), _candidate(1, binding=[1], soft=4)])
    assert soft_order["selected_candidate_id"] == "candidate_1"


def test_scenario_constraint_6813_stable_tie_uses_frozen_order() -> None:
    """SCENARIO-CONSTRAINT-6813-STABLE-TIE rejects outcome-dependent ties."""

    candidates = [_candidate(0, binding=[1], soft=7), _candidate(1, binding=[1], soft=7)]
    first = _select(candidates)
    second = _select(list(reversed(candidates)))

    assert first["selected_candidate_id"] == "candidate_0"
    assert second["selected_candidate_id"] == "candidate_0"
    assert first["selected_action_bytes_b64"] == second["selected_action_bytes_b64"]


def test_scenario_constraint_6813_no_legal_candidate_abstains() -> None:
    """SCENARIO-CONSTRAINT-6813-NO-LEGAL returns fallback and a certificate."""

    decision = _select(
        [
            _candidate(0, parse_state="incomplete", legal=False),
            _candidate(1, hard=1, legal=False, first_conflict="hard-1"),
        ]
    )

    assert decision["selected_candidate_id"] is None
    assert decision["abstention"] is True
    assert decision["selected_action"] == {"data": None, "kind": "NOOP"}
    assert decision["certificate"]["kind"] == "no_legal_candidate"
    assert decision["certificate_complete"] is True


def test_scenario_constraint_6813_no_op_preserves_explicit_safe_base_bytes() -> None:
    """SCENARIO-CONSTRAINT-6813-NO-OP preserves a valid input base exactly."""

    base = {"data": {"preserve": "these bytes"}, "kind": "SAFE"}
    decision = _select(
        [_candidate(0, soft=1), _candidate(1, soft=99)],
        base_action=base,
        base_valid=True,
    )

    assert decision["selected_candidate_id"] == "base_proposal"
    assert decision["selected_action_bytes_b64"] == exp.b64_bytes(exp.canonical_json_bytes(base))
    assert decision["false_intervention"] is False
    assert decision["safe_action_identity"] is True
    assert decision["certificate"] == {
        "kind": "no_op_preserved",
        "first_higher_priority_conflict": None,
        "rejections": [],
    }


def test_scenario_constraint_6813_conflict_certificate_uses_first_priority() -> None:
    """SCENARIO-CONSTRAINT-6813-CERTIFICATE names syntax, hard, then binding."""

    parse = exp.candidate_conflict(_candidate(0, parse_state="incomplete", legal=False))
    hard = exp.candidate_conflict(
        _candidate(0, hard=1, binding=[1], legal=False, first_conflict="hard-first")
    )
    binding = exp.candidate_conflict(_candidate(0, binding=[0, 1], soft=-999))

    assert parse["first_conflict"] == "response_schema"
    assert hard["first_conflict"] == "hard-first"
    assert binding["first_conflict"] == "binding-1"
    assert [parse["priority"], hard["priority"], binding["priority"]] == [
        "syntax",
        "hard",
        "binding",
    ]


def test_scenario_constraint_6813_flat_retry_and_false_intervention() -> None:
    """REQ-CONSTRAINT-6813 keeps flat order and counts only safe-base changes."""

    base = {"data": "original", "kind": "SAFE"}
    flat = _select(
        [_candidate(0, hard=1, legal=False), _candidate(1)],
        arm=exp.FLAT_ARM,
        base_action=base,
        base_valid=True,
    )

    assert flat["selected_candidate_id"] == "candidate_1"
    assert flat["retry_count"] == 1
    assert flat["false_intervention"] is True
    assert flat["safe_action_identity"] is False

    rate = exp.false_intervention_metric(
        [
            {"base_already_valid": True, "false_intervention": True},
            {"base_already_valid": True, "false_intervention": False},
            {"base_already_valid": False, "false_intervention": True},
        ],
        alpha=0.05,
    )
    assert rate["numerator"] == 1
    assert rate["denominator"] == 2
    assert rate["rate"] == 0.5
    assert rate["upper_bound"] > rate["rate"]


def test_scenario_constraint_6813_budget_parity_detects_mismatch() -> None:
    """SCENARIO-CONSTRAINT-6813-ACCOUNTING requires equal pair budgets."""

    common = {
        "split": "held",
        "candidate_count": 2,
        "exact_check_count": 2,
        "outcome_check_count": 1,
        "work_units": 3,
        "retry_cap": 1,
        "cpu_allowance_us": 100_000,
        "latency_us": 10.0,
    }
    rows = [
        {**common, "pair_id": "pair", "arm": exp.SELECTIVE_ARM},
        {**common, "pair_id": "pair", "arm": exp.FLAT_ARM},
    ]
    receipt = exp.derive_budget_match_receipt(rows)
    assert receipt["passed"] is True
    assert receipt["equal_observed_work"] is True

    drift = deepcopy(rows)
    drift[1]["exact_check_count"] = 1
    mismatch = exp.derive_budget_match_receipt(drift)
    assert mismatch["passed"] is False
    assert mismatch["mismatched_pair_ids"] == ["pair"]


def test_req_constraint_6813_paired_interval_and_positive_gate() -> None:
    """REQ-CONSTRAINT-6813 uses held pairs and keeps completion effect-independent."""

    rows: list[dict] = []
    for index in range(20):
        common = {"pair_id": f"p{index}", "split": "held"}
        rows.extend(
            [
                {
                    **common,
                    "arm": exp.SELECTIVE_ARM,
                    "accepted_progress": 1,
                    "retry_count": 0,
                },
                {
                    **common,
                    "arm": exp.FLAT_ARM,
                    "accepted_progress": 0,
                    "retry_count": 1,
                },
            ]
        )
    progress, retry = exp.derive_paired_deltas(rows, seed=681302, resamples=200)

    assert progress["estimate"] == 1.0
    assert progress["lower_bound"] == 1.0
    assert retry["estimate"] == 1.0
    assert retry["direction"] == "flat_minus_selective"

    gate = exp.derive_acceptance_gate(
        selective_hard_rate=0.0,
        selective_false_upper=0.1,
        false_limit=0.2,
        progress_lower=progress["lower_bound"],
        retry_lower=retry["lower_bound"],
        no_family_support_loss=True,
        selective_harmful=0,
        flat_harmful=0,
    )
    assert gate["passed"] is True
    assert all(gate["conditions"].values())


@pytest.fixture(scope="module")
def built_artifact() -> dict:
    return exp.build_artifact(run_date="20260831", duration_s=1.25)


def test_req_constraint_6813_builds_complete_authentic_ab_rows(built_artifact: dict) -> None:
    """REQ-CONSTRAINT-6813 replays every authentic source cell in both arms."""

    artifact = built_artifact
    assert artifact["selective_arbiter_ab_completed"] is True
    assert artifact["source_artifact_sha256"].startswith("sha256:")
    assert len(artifact["rows"]) == 3 * 48 * 2 * 2
    assert {row["arm"] for row in artifact["rows"]} == set(exp.ARMS)
    assert {row["split"] for row in artifact["rows"]} == {"development", "held"}
    assert all(row["exact_check_count"] == 2 for row in artifact["rows"])
    assert all(row["outcome_check_count"] == 1 for row in artifact["rows"])
    assert artifact["budget_match_receipt"]["passed"] is True
    assert artifact["hard_violation_rate_by_arm"][exp.SELECTIVE_ARM]["rate"] == 0.0
    assert artifact["false_intervention_rate_by_arm"][exp.SELECTIVE_ARM]["rate"] == 0.0
    assert artifact["safe_action_identity_by_arm"][exp.SELECTIVE_ARM]["rate"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in {"positive", "null"}
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []


def test_req_constraint_6813_manifest_constants_and_features_are_frozen(
    built_artifact: dict,
) -> None:
    """REQ-CONSTRAINT-6813 freezes split and feature authority before held reduction."""

    manifest = built_artifact["frozen_manifest"]
    assert len(manifest["development_scenario_ids"]) == 24
    assert len(manifest["held_scenario_ids"]) == 24
    assert set(manifest["development_scenario_ids"]).isdisjoint(manifest["held_scenario_ids"])
    assert manifest["public_constants"]["fit_split"] == "development"
    assert "model_id" in built_artifact["feature_denylist"]
    assert "exact_utility" in built_artifact["feature_denylist"]
    assert "hard_violation_count" in built_artifact["feature_allowlist"]
    assert all(result["passed"] for result in built_artifact["attack_results"])


def test_req_constraint_6813_checksum_and_validator_detect_drift(
    built_artifact: dict,
) -> None:
    """REQ-CONSTRAINT-6813 binds deterministic rows but excludes measured clocks."""

    timing_only = deepcopy(built_artifact)
    timing_only["duration_s"] = 99.0
    timing_only["rows"][0]["latency_us"] += 123.0
    assert exp.reproducibility_checksum(timing_only) == built_artifact["reproducibility_checksum"]

    drift = deepcopy(built_artifact)
    drift["paired_retry_delta"]["estimate"] += 1.0
    errors = exp.validate_artifact(drift)
    assert "paired retry delta is not row-derived" in errors
    assert "reproducibility checksum mismatch" in errors


def test_req_constraint_6813_blocked_preconditions_stop_before_rows(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6813 writes the named blocked shape without comparison rows."""

    artifact = exp.build_artifact(repo_root=tmp_path, run_date="20260831", duration_s=0.5)

    assert artifact["status"] == exp.BLOCKED_STATUS
    assert artifact["honest_verdict"] == exp.BLOCKED_STATUS
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["selective_arbiter_ab_completed"] is False
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_checks"]
    assert exp.validate_artifact(artifact) == []


def test_req_constraint_6813_feature_attack_cannot_change_selection() -> None:
    """REQ-CONSTRAINT-6813 denies identity and outcome labels as features."""

    candidates = [_candidate(0, binding=[1], soft=1), _candidate(1, binding=[0], soft=0)]
    clean = _select(candidates)
    attacked = deepcopy(candidates)
    attacked[0].update(
        {
            "model_id": "preferred-family",
            "exact_utility": 10**30,
            "future_outcome": "win",
            "harmful_selection": False,
        }
    )
    attacked[1].update(
        {
            "model_id": "other-family",
            "exact_utility": -(10**30),
            "future_outcome": "loss",
            "harmful_selection": True,
        }
    )

    assert _select(attacked)["selected_candidate_id"] == clean["selected_candidate_id"]


def test_req_constraint_6813_defensive_selection_and_empty_reducers() -> None:
    """REQ-CONSTRAINT-6813 fails closed on authority, arms, and empty evidence."""

    authority = _candidate(0)
    authority["authority_preserved"] = False
    assert exp.candidate_conflict(authority)["first_conflict"] == "authority_spoof"

    with pytest.raises(ValueError, match="unknown arm"):
        _select([_candidate(0)], arm="oracle_arm")

    empty_false = exp.false_intervention_metric([], alpha=0.05)
    assert empty_false["upper_bound"] is None
    progress, retry = exp.derive_paired_deltas([], seed=681302, resamples=10)
    assert progress["estimate"] is None
    assert retry["estimate"] is None
