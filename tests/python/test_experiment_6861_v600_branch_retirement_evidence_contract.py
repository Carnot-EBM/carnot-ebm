"""Tests for the immutable V600 evidence root.

Spec refs: REQ-REPORT-6861 and SCENARIO-REPORT-6861-*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6861_v600_branch_retirement_evidence_contract as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _one_source_spec(path: str, frozen_sha256: str) -> dict[str, dict[str, object]]:
    return {
        "exp-test": {
            "task_id": "exp-test",
            "title_prefix": "Test task",
            "path": path,
            "branch": "test",
            "frozen_sha256": frozen_sha256,
            "gate_record_allowed": False,
        }
    }


def test_missing_source_inventory_blocks_readiness(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6861-MISSING-SOURCE
    specs = _one_source_spec("results/missing.json", "sha256:" + "0" * 64)
    inventory = mod.inspect_source_inventory(tmp_path, specs, conductor_text="")

    assert inventory["missing_artifact_inventory"] == ["results/missing.json"]
    assert inventory["preconditions_passed"] is False
    assert inventory["gate_check_summary"]["failed_check"] == "terminal_source_inventory"
    assert inventory["payloads"] == {}


def test_explicit_gate_record_satisfies_terminal_inventory(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6861-MISSING-SOURCE
    specs = _one_source_spec("results/gated.json", "sha256:" + "0" * 64)
    specs["exp-test"]["gate_record_allowed"] = True
    events = "| 2026-09-01 20:56 UTC | Test task | GATE_BLOCK | missing headroom |"

    inventory = mod.inspect_source_inventory(tmp_path, specs, conductor_text=events)

    assert inventory["missing_artifact_inventory"] == []
    assert inventory["conductor_skip_manifest"][0]["task_id"] == "exp-test"
    assert inventory["preconditions_passed"] is True


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6861-SOURCE-DRIFT
    source = tmp_path / "results/source.json"
    _write_json(source, {"status": "complete"})
    specs = _one_source_spec("results/source.json", "sha256:" + "f" * 64)

    inventory = mod.inspect_source_inventory(tmp_path, specs, conductor_text="")

    assert inventory["source_drift_manifest"][0]["source_id"] == "exp-test"
    assert inventory["preconditions_passed"] is False
    assert inventory["gate_check_summary"]["failed_check"] == "immutable_source_hashes"


def test_malformed_source_is_a_complete_missing_record(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6861-MISSING-SOURCE
    source = tmp_path / "results/source.json"
    source.parent.mkdir(parents=True)
    source.write_text("[]", encoding="utf-8")
    digest = mod.sha256_path(source)
    inventory = mod.inspect_source_inventory(
        tmp_path,
        _one_source_spec("results/source.json", digest),
        conductor_text="",
    )

    assert inventory["missing_artifact_inventory"] == ["results/source.json"]
    assert mod.read_json(tmp_path / "does-not-exist.json") is None


def test_flagged_sources_quarantine_claims_but_preserve_receipts() -> None:
    # SCENARIO-REPORT-6861-FLAGGED-UPSTREAM
    dispositions = mod.recompute_dispositions(
        {
            "exp6856": {
                "rows": [
                    {
                        "arm": "contextual_bandit",
                        "split": "held_future",
                        "effect_vs_no_memory": 0.2,
                        "abstained": True,
                    }
                ],
                "restart_results": {"byte_identical": True},
                "rollback_results": {"restored_parent_bytes": True},
            },
            "exp6855": {
                "rows": [
                    {
                        "counterfactual_metric": "marginal_value",
                        "write_id": "write-1",
                        "metric_value": -1.0,
                    }
                ]
            },
            "exp6859": {
                "rows": [
                    {
                        "row_kind": "gap_chain",
                        "join_complete": True,
                        "first_party": True,
                        "provenance_class": "fixture",
                    }
                ]
            },
        },
        flagged_source_ids={"exp6856"},
    )

    assert dispositions["self_learning"]["verdict_class"] == "disqualified"
    assert dispositions["self_learning"]["scientific_claim_eligible"] is False
    assert dispositions["self_learning"]["transaction_infrastructure_reusable"] is True
    assert dispositions["tool_gap"]["receipt_contract_ready"] is True


def test_procedural_positive_capstone_does_not_promote_scientific_branches() -> None:
    # SCENARIO-REPORT-6861-PROCEDURAL-POSITIVE
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    branches = {row["branch"]: row for row in artifact["terminal_branch_manifest"]}

    assert artifact["v600_evidence_contract_ready_score"] == 1
    assert branches["typed_authority"]["verdict_class"] == "positive"
    assert branches["compatibility"]["verdict_class"] == "null"
    assert branches["self_learning"]["verdict_class"] == "disqualified"
    assert branches["supervisor"]["verdict_class"] == "blocked"
    assert branches["tool_gap"]["verdict_class"] == "partial"
    assert branches["v599_capstone"]["scope"] == "procedural_only"
    assert branches["v599_capstone"]["scientific_branch_advance_count"] == 0


def test_failed_mechanism_reuse_requires_a_real_delta_and_gate() -> None:
    # SCENARIO-REPORT-6861-FAILED-MECHANISM-REUSE
    retired = {"raw_fixed_sequence_margin_claim"}
    invalid = mod.validate_changed_mechanisms(
        [
            {
                "mechanism_id": "raw_fixed_sequence_margin_claim",
                "reuses_retired_mechanism": True,
                "method_delta": "",
                "falsifiable_gate": "",
            }
        ],
        retired,
    )

    assert invalid == [
        {
            "mechanism_id": "raw_fixed_sequence_margin_claim",
            "reason": "retired_mechanism_reused_without_method_delta_and_gate",
        }
    ]
    assert mod.validate_changed_mechanisms(mod.changed_mechanism_manifest(), retired) == []


def test_split_contract_names_every_downstream_gate_exactly() -> None:
    # REQ-REPORT-6861
    contract = mod.build_split_and_gate_contract()

    assert contract["calibration_split_hash_field"] == "calibration_split_sha256"
    assert contract["held_split_hash_field"] == "held_split_sha256"
    assert contract["split_disjointness_field"] == "calibration_held_group_overlap_count"
    assert contract["downstream_gate_fields"] == [
        {
            "consumer": "exp6862-dual-side-semantic-contrast-bank",
            "artifact_field": "v600_evidence_contract_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "consumer": "exp6867-decision-time-observability-firewall",
            "artifact_field": "v600_evidence_contract_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "consumer": "exp6870-resumable-live-arc-receipt-checkpoint-harness",
            "artifact_field": "v600_evidence_contract_ready_score",
            "op": "==",
            "value": 1,
        },
    ]
    assert "live_receipt_identity" in contract["provenance_schema"]["required_fields"]
    assert "process_owner_pid" in contract["provenance_schema"]["required_fields"]


def test_reference_rows_match_primary_pages_and_record_planner_drift() -> None:
    # REQ-REPORT-6861
    rows = mod.reference_verification_rows()

    assert {row["arxiv_id"] for row in rows} == {
        "2606.10616",
        "2605.29556",
        "2604.09459",
        "2608.15008",
    }
    assert all(row["primary_page_verified"] for row in rows)
    assert all(row["method_delta"] and row["carnot_hook"] for row in rows)
    assert all(row["access_boundary"] for row in rows)
    credit = next(row for row in rows if row["arxiv_id"] == "2604.09459")
    opt = next(row for row in rows if row["arxiv_id"] == "2605.29556")
    assert credit["primary_page_facts"]["paper_count"] == 69
    assert credit["planner_metadata_correction"]["recorded_method_count"] == 47
    assert opt["primary_page_facts"]["venue"] == "ICML 2026"


def test_blocked_artifact_keeps_full_schema_and_exact_failure(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6861-MISSING-SOURCE
    artifact = mod.build_artifact(
        tmp_path,
        "20260902",
        source_specs=_one_source_spec("results/missing.json", "sha256:" + "0" * 64),
        conductor_text="",
    )

    assert artifact["v600_evidence_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_v600_branch_retirement_evidence_contract"
    assert artifact["gate_check_summary"]["failed_check"] == "terminal_source_inventory"
    assert mod.validate_artifact(artifact) == []


def test_artifact_schema_checksum_and_rows_are_self_consistent() -> None:
    # REQ-REPORT-6861
    artifact = mod.build_artifact(REPO_ROOT, "20260902")

    assert mod.validate_artifact(artifact) == []
    assert artifact["inference_substrate"] == "deterministic CPU evidence replay"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    row_kinds = {row["row_kind"] for row in artifact["rows"]}
    assert {"source", "disposition", "retirement", "reference"} <= row_kinds


def test_validator_reports_all_material_schema_errors() -> None:
    # REQ-REPORT-6861
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    artifact.pop("rows")
    artifact["inference_substrate"] = "wrong"
    artifact["verifier_is_oracle"] = True
    artifact["verdict_class"] = "unknown"
    artifact["honest_verdict"] = "not-terminal"
    artifact["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(artifact)

    assert "missing_required_fields:rows" in errors
    assert "invalid_inference_substrate" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_atomic_writer_and_cli_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-6861
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    output = tmp_path / "artifact.json"
    mod.write_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"].startswith("exp6861")

    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: artifact)
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 0


def test_cli_refuses_invalid_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-6861
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    artifact["verifier_is_oracle"] = True
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: artifact)

    with pytest.raises(ValueError, match="verifier_is_oracle_must_be_false"):
        mod.main(["--date", "20260902", "--output", str(tmp_path / "bad.json")])


def test_new_reducer_never_imports_the_exp6860_reducer() -> None:
    # REQ-REPORT-6861
    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert "experiment_6860_v599_independent_capstone import" not in source
    assert "producer_aggregate_functions_imported" in source
