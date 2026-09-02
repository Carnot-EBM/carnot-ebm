"""Tests for the prospective sealed self-learning audit.

Spec refs: REQ-LEARN-6873 and SCENARIO-LEARN-6873-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_6873_prospective_sealed_self_learning_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _event(
    identity: str,
    *,
    bucket: int,
    direction: int = 1,
    delayed: bool = False,
    stale: bool = False,
) -> dict[str, object]:
    """Build one event whose outcome stays outside the decision features."""

    content = f"sha256:{'0' * 62}{bucket:02x}"
    return {
        "event_identity": identity,
        "primary_content_sha256": exp.sha256_json(f"primary:{identity}"),
        "primary_source": {
            "transaction_receipt_sha256": exp.sha256_json(f"receipt:{identity}"),
        },
        "decision_features": {
            "source_content_sha256": content,
            "family": "old-family",
            "evidence_status_at_decision": "stale" if stale else "fresh",
        },
        "counterfactual": {"valid_pre_action": True},
        "action_support": {"bounded_update": {"supported": not stale}},
        "later_exact_outcome": {
            "signed_direction": direction,
            "exact_outcome_hash": exp.sha256_json(f"outcome:{identity}"),
            "source_row_sha256": exp.sha256_json(f"outcome-row:{identity}"),
            "source_artifact": "outcomes_a",
            "revealed_after_decision": True,
            "delayed_correction": {
                "correction_family": "invalidated" if delayed else None,
                "correction_latency_events": 2 if delayed else 0,
            },
        },
    }


def _events() -> list[dict[str, object]]:
    """Exercise every controller action and one delayed correction."""

    return [
        _event("write-useful", bucket=0),
        _event("read", bucket=3),
        _event("none", bucket=5, direction=-1),
        _event("abstain", bucket=6),
        _event("write-corrected", bucket=1, delayed=True),
    ]


def _source(events: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Build five frozen orders over one compact event roster."""

    rows = events or _events()
    identities = [str(row["event_identity"]) for row in rows]
    orders = [identities, identities[1:] + identities[:1], list(reversed(identities))]
    orders.extend([identities[2:] + identities[:2], identities[3:] + identities[:3]])
    return {
        "rows": rows,
        "counterfactual_support_rows": [
            {"event_identity": identity, "valid_pre_action": True} for identity in identities
        ],
        "order_replicate_manifest": [
            {
                "replicate_id": f"order_replicate_{index}",
                "seed": 6_871_010 + index,
                "event_identities": order,
                "all_events_preserved": True,
            }
            for index, order in enumerate(orders, start=1)
        ],
    }


def _write_inputs(root: Path) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    """Write an isolated controller and stream with exact fixture hashes."""

    source_path = root / exp.SOURCE_RELATIVE_PATH
    controller_path = root / exp.CONTROLLER_RELATIVE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    fixture_files: dict[str, Path] = {}
    for key in ("controller_module", "controller_wrapper", "controller_tests"):
        path = root / f"frozen/{key}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key, encoding="utf-8")
        fixture_files[key] = path
    expected = {
        "opportunity_stream": {
            "path": str(source_path.relative_to(root)),
            "sha256": exp.sha256_file(source_path),
        },
        **{
            key: {
                "path": str(path.relative_to(root)),
                "sha256": exp.sha256_file(path),
            }
            for key, path in fixture_files.items()
        },
    }
    controller = {
        "bounded_reliability_controller_ready_score": 1,
        "no_model_weight_mutation": True,
        "counterfactual_support_rows": _source()["counterfactual_support_rows"],
        "order_seed_manifest": [
            {
                "replicate_id": row["replicate_id"],
                "seed": row["seed"],
                "all_events_preserved": True,
            }
            for row in _source()["order_replicate_manifest"]
        ],
        "source_artifact_hashes": {
            "observable_reliability_stream": expected["opportunity_stream"],
            "module": expected["controller_module"],
            "wrapper": expected["controller_wrapper"],
            "focused_tests": expected["controller_tests"],
        },
        "model_immutability_receipt": {
            "before_sha256": exp.sha256_json("no-model"),
            "after_sha256": exp.sha256_json("no-model"),
        },
    }
    controller_path.write_text(json.dumps(controller), encoding="utf-8")
    expected["controller_artifact"] = {
        "path": str(controller_path.relative_to(root)),
        "sha256": exp.sha256_file(controller_path),
    }
    return controller_path, source_path, expected


def _order_metric(
    replicate: str,
    arm: str,
    utility: float,
    *,
    actions: int = 2,
    useful: int = 1,
    harmful: int = 0,
    retention: float = 1.0,
) -> dict[str, object]:
    """Build one row-reduced order metric for claim-gate tests."""

    return {
        "replicate_id": replicate,
        "arm": arm,
        "row_count": 4,
        "distinct_action_count": actions,
        "action_entropy": 1.0 if actions > 1 else 0.0,
        "abstention_rate": 0.25,
        "admitted_useful_updates": useful,
        "harmful_writes": harmful,
        "false_injection_rate": harmful / 4,
        "held_future_utility_mean": utility,
        "held_future_utility_sum": utility * 2,
        "old_family_retention_rate": retention,
    }


def _per_order(
    *,
    quarantine_utility: float = 0.6,
    frozen_utility: float = 0.2,
    read_utility: float = 0.1,
) -> list[dict[str, object]]:
    """Build five complete order blocks with a strict quarantine win."""

    rows: list[dict[str, object]] = []
    for index in range(1, 6):
        replicate = f"order_replicate_{index}"
        rows.extend(
            [
                _order_metric(replicate, "frozen_no_memory", frozen_utility, actions=1, useful=0),
                _order_metric(replicate, "read_only", read_utility, useful=0),
                _order_metric(replicate, "bounded_update", 0.5),
                _order_metric(replicate, "exact_quarantine", quarantine_utility),
                _order_metric(
                    replicate,
                    "v599_unsafe_reference",
                    -0.4,
                    actions=1,
                    harmful=1,
                    retention=0.75,
                ),
            ]
        )
    return rows


def _safe_audits() -> dict[str, object]:
    """Build complete durability evidence for utility-gate unit tests."""

    return {
        "spectral_bound_audit_rows": [
            {
                "arm": "exact_quarantine",
                "within_bound": True,
                "state_symmetric_before": True,
                "state_symmetric_after": True,
            }
        ],
        "delayed_correction_rows": [{"arm": "exact_quarantine", "passed": True}],
        "persistence_rows": [{"byte_exact": True, "private_checkpoint": True}],
        "restart_rows": [
            {
                "byte_exact": True,
                "state_restored": True,
                "memory_restored": True,
                "tombstones_restored": True,
                "fresh_process": True,
            }
        ],
        "rollback_rows": [{"byte_exact": True}],
        "leakage_witnesses": [],
        "no_model_weight_mutation": True,
        "aggregate_consistent": True,
        "process_isolation_passed": True,
    }


def test_one_order_only_win_and_no_headroom_cannot_authorize_positive() -> None:
    # SCENARIO-LEARN-6873-ONE-ORDER
    # SCENARIO-LEARN-6873-NO-HEADROOM
    per_order = _per_order()
    per_order[3]["held_future_utility_mean"] = 0.2
    paired = exp.compute_paired_order_effects(per_order)

    comparison = paired["exact_quarantine_vs_frozen_no_memory"]
    assert comparison["wins"] == 4
    assert comparison["ties"] == 1
    assert comparison["declared_replication_rule_passed"] is False

    saturated = _per_order(quarantine_utility=1.0, frozen_utility=1.0)
    saturated_paired = exp.compute_paired_order_effects(saturated)
    saturated_comparison = saturated_paired["exact_quarantine_vs_frozen_no_memory"]
    assert saturated_comparison["available_headroom_by_order"] == {
        f"order_replicate_{index}": 0.0 for index in range(1, 6)
    }
    assert saturated_comparison["declared_replication_rule_passed"] is False


def test_action_collapse_and_zero_admitted_writes_return_null() -> None:
    # SCENARIO-LEARN-6873-ACTION-COLLAPSE
    # SCENARIO-LEARN-6873-ZERO-WRITES
    collapsed = _per_order()
    for row in collapsed:
        if row["arm"] == "exact_quarantine":
            row["distinct_action_count"] = 1
            row["action_entropy"] = 0.0
    verdict = exp.evaluate_claim(
        collapsed, exp.compute_paired_order_effects(collapsed), _safe_audits()
    )
    assert verdict["verdict_class"] == "null"
    assert verdict["checks_by_name"]["quarantine_action_not_collapsed"] is False

    zero_writes = _per_order()
    for row in zero_writes:
        if row["arm"] == "exact_quarantine":
            row["admitted_useful_updates"] = 0
    verdict = exp.evaluate_claim(
        zero_writes, exp.compute_paired_order_effects(zero_writes), _safe_audits()
    )
    assert verdict["verdict_class"] == "null"
    assert verdict["checks_by_name"]["quarantine_admitted_useful_write"] is False


def test_harm_forgetting_state_restart_rollback_and_leakage_disqualify() -> None:
    # SCENARIO-LEARN-6873-HARMFUL-WRITE
    # SCENARIO-LEARN-6873-FORGETTING
    # SCENARIO-LEARN-6873-STATE-BOUND
    # SCENARIO-LEARN-6873-RESTART
    # SCENARIO-LEARN-6873-ROLLBACK
    # SCENARIO-LEARN-6873-LEAKAGE
    cases = (
        ("harm", lambda orders, audits: orders[3].update(harmful_writes=1)),
        ("forget", lambda orders, audits: orders[3].update(old_family_retention_rate=0.75)),
        (
            "state",
            lambda orders, audits: audits["spectral_bound_audit_rows"][0].update(
                within_bound=False
            ),
        ),
        ("restart", lambda orders, audits: audits["restart_rows"][0].update(byte_exact=False)),
        ("rollback", lambda orders, audits: audits["rollback_rows"][0].update(byte_exact=False)),
        (
            "leakage",
            lambda orders, audits: audits["leakage_witnesses"].append({"field": "outcome"}),
        ),
    )
    for _name, mutate in cases:
        orders = _per_order()
        audits = _safe_audits()
        mutate(orders, audits)
        verdict = exp.evaluate_claim(orders, exp.compute_paired_order_effects(orders), audits)
        assert verdict["verdict_class"] == "disqualified"
        assert verdict["scientific_claim_eligible"] is False


def test_delayed_correction_is_tombstoned_and_rollback_is_exact(tmp_path: Path) -> None:
    # SCENARIO-LEARN-6873-DELAYED-CORRECTION
    # SCENARIO-LEARN-6873-ROLLBACK
    checkpoint = tmp_path / "checkpoint.json"
    result = exp.run_arm_replicate(
        [_event("corrected", bucket=0, delayed=True)],
        "exact_quarantine",
        "order_replicate_1",
        6_871_011,
        checkpoint,
    )

    row = result["rows"][0]
    assert row["admission_decision"] == "quarantined"
    assert row["delayed_correction"]["passed"] is True
    assert row["tombstone_evidence"]["present"] is True
    assert row["rollback_evidence"]["byte_exact"] is True
    assert result["checkpoint"]["memory"] == []
    assert result["checkpoint"]["tombstones"][0]["event_identity"] == "corrected"

    restored = exp.restore_checkpoint(checkpoint)
    assert restored["byte_exact"] is True
    assert restored["tombstones_restored"] is True


def test_unsafe_reference_failure_does_not_self_disqualify_safe_quarantine() -> None:
    # SCENARIO-LEARN-6873-DELAYED-CORRECTION
    orders = _per_order()
    audits = _safe_audits()
    audits["delayed_correction_rows"].append({"arm": "v599_unsafe_reference", "passed": False})

    verdict = exp.evaluate_claim(orders, exp.compute_paired_order_effects(orders), audits)
    assert verdict["verdict_class"] == "positive"


def test_failed_frozen_hash_precondition_emits_complete_blocked_shape(tmp_path: Path) -> None:
    # SCENARIO-LEARN-6873-PRECONDITIONS
    controller, source, expected = _write_inputs(tmp_path)
    expected["controller_module"]["sha256"] = exp.sha256_json("wrong")
    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        controller_relative_path=controller,
        source_relative_path=source,
        expected_hashes=expected,
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "exact_frozen_source_and_code_hashes"
    assert exp.validate_artifact(artifact) == []


def test_five_orders_run_in_private_fresh_processes(tmp_path: Path) -> None:
    # REQ-LEARN-6873
    # SCENARIO-LEARN-6873-FRESH-PROCESS
    controller, source, expected = _write_inputs(tmp_path)
    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        controller_relative_path=controller,
        source_relative_path=source,
        expected_hashes=expected,
    )

    assert len(artifact["rows"]) == 5 * 5 * 5
    isolation = artifact["process_isolation_rows"]
    assert len(isolation) == 25
    assert len({row["worker_process_id"] for row in isolation}) == 25
    assert all(row["worker_process_id"] != row["restart_process_id"] for row in isolation)
    assert all(row["initial_state_matches_frozen"] for row in isolation)
    assert all(row["initial_memory_empty"] and row["initial_cache_empty"] for row in isolation)
    assert all(row["private_checkpoint"] for row in isolation)
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert exp.validate_artifact(artifact) == []


def test_aggregate_row_contradiction_disqualifies_validator() -> None:
    # SCENARIO-LEARN-6873-AGGREGATE-CONTRADICTION
    rows = [
        {
            "replicate_id": "order_replicate_1",
            "arm": "exact_quarantine",
            "proposed_action": "write",
            "admission_decision": "admitted",
            "useful_write": True,
            "harmful_write": False,
            "false_injection": False,
            "held_future": True,
            "utility": 1.0,
            "anchor_retained": True,
        }
    ]
    headlines = exp.recompute_headlines(rows)
    artifact = exp.empty_artifact("20260902")
    artifact.update(headlines)
    artifact["rows"] = rows
    artifact["preconditions_checked"] = {"passed": True, "checks": []}
    artifact["gate_check_summary"] = exp.gate_summary([])
    artifact["verdict_class"] = "disqualified"
    artifact["honest_verdict"] = exp.DISQUALIFIED_VERDICT
    artifact["harmful_writes_by_arm"]["exact_quarantine"] = 1
    artifact["field_principles"] = exp.field_principles(artifact)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)

    errors = exp.validate_artifact(artifact)
    assert "aggregate_row_contradiction:harmful_writes_by_arm" in errors


def test_cli_writes_blocked_artifact_without_touching_repository(tmp_path: Path) -> None:
    # REQ-LEARN-6873
    output = tmp_path / "blocked.json"
    missing_controller = tmp_path / "missing-controller.json"
    missing_source = tmp_path / "missing-source.json"

    assert (
        exp.main(
            [
                "--date",
                "20260902",
                "--root",
                str(tmp_path),
                "--controller",
                str(missing_controller),
                "--source",
                str(missing_source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "controller_artifact_readable"


def test_checked_in_artifact_is_complete_and_row_authoritative() -> None:
    # REQ-LEARN-6873
    path = REPO_ROOT / exp.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert exp.validate_artifact(artifact) == []
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert len(artifact["rows"]) == 765 * 5 * 5
    assert artifact["verifier_is_oracle"] is False
    assert artifact["honest_verdict"].startswith("complete_")
