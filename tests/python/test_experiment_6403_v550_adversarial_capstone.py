"""Tests for Exp6403 V550 adversarial capstone.

Spec refs: REQ-CAPSTONE-6403, SCENARIO-CAPSTONE-6403,
SCENARIO-CAPSTONE-6403-DECISIONS,
SCENARIO-CAPSTONE-6403-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6403_v550_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _receipts() -> list[dict[str, object]]:
    return [{"command": mod.FOCUSED_TEST_COMMAND, "exit_code": 0}]


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260813",
        command_receipts=_receipts(),
        e2e_receipts=[{"command": mod.E2E_PLAN_READ_COMMAND, "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        duration_s=1.0,
    )


def test_req_capstone_6403_spec_declares_required_schema() -> None:
    """REQ-CAPSTONE-6403: OpenSpec owns the V550 capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6403") :]

    for token in (
        "SCENARIO-CAPSTONE-6403",
        "SCENARIO-CAPSTONE-6403-DECISIONS",
        "SCENARIO-CAPSTONE-6403-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "universal support false",
        "Public claim eligibility SHALL remain false",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_capstone_6403_preserves_classes_and_hashes() -> None:
    """SCENARIO-CAPSTONE-6403: evidence classes and protected hashes stay visible."""

    report = _report()
    matrix = report[
        "present_absent_blocked_skipped_null_partial_abstained_flagged_retired_and_clean_matrix"
    ]

    assert mod.validate_report(report) == []
    assert report["expected_task_ids_and_deliverables"]["expected_task_ids"] == list(
        mod.EXPECTED_TASK_IDS
    )
    assert matrix["classification_before_decisions"] is True
    assert matrix["class_counts"]["clean"] >= 1
    assert matrix["class_counts"]["null"] >= 1
    assert matrix["cell_state_counts"]["abstained"] == 3
    assert matrix["cell_state_counts"]["rejected"] == 2
    assert matrix["cell_state_counts"]["licensed"] == 4
    assert matrix["by_task"]["exp6399"]["terminal_class"] == "null"
    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["solve_registry_modified"] is False
    assert report["claims_ledger_modified"] is False
    assert report["protected_files_unchanged"]["ok"] is True


def test_scenario_capstone_6403_recomputes_gates_and_factor_decisions() -> None:
    """SCENARIO-CAPSTONE-6403-DECISIONS: factor gates fail closed and stay scoped."""

    report = _report()
    gates = report["recomputed_gate_type_finiteness_identity_hash_and_principle_checks"]
    licenses = report["factor_harness_license_and_universal_support_decision"]
    frontier = report["factor_frontier_alignment_learnability_and_future_utility_decision"]
    learning = report["transactional_continuous_self_learning_decision"]
    consumer = report["rollback_and_consumer_decision"]

    assert gates["all_recomputed_gates_passed"] is True
    assert gates["by_gate"]["exp6395.held_factor_transport_license_ready_score"]["passed"] is True
    assert gates["by_gate"]["exp6395.universal_support_claimed"]["passed"] is True
    assert licenses["held_license_ready"] is True
    assert licenses["licensed_model_count"] == 2
    assert licenses["licensed_constraint_family_count"] == 3
    assert licenses["universal_support"] == "false_partial_capability"
    assert frontier["decision"] == "positive_scoped_to_licensed_cells"
    assert frontier["delta_verified_future_exact_yield"] > 0
    assert learning["decision"] == "partial_fr11_evidence_positive_but_not_public"
    assert learning["transaction_dispositions"] == {
        "Commit": 2,
        "Defer": 1,
        "Quarantine": 1,
        "Reject": 2,
    }
    assert consumer["consumer_decision"] == "default_off_positive_internal_only"
    assert consumer["production_enable_count"] == 0


def test_scenario_capstone_6403_recomputes_arc_and_public_boundaries() -> None:
    """SCENARIO-CAPSTONE-6403-DECISIONS: ARC promotion is internal and no-solve."""

    report = _report()

    assert report["arc_scalar_contract_decision"]["decision"] == "ready"
    assert (
        report["arc_shadow_reachability_and_provenance_decision"]["decision"]
        == "shadow_reachable_default_off_no_solve"
    )
    causal = report["arc_causal_progress_false_accept_and_oracle_timing_decision"]
    assert causal["decision"] == "route_promotion_internal_only"
    assert causal["delta_exact_progress_proxy"] > 0
    assert causal["delta_false_accept_count"] <= 0
    assert causal["oracle_timing_passed"] is True
    audit = report["arc_safety_audit_and_no_solve_decision"]
    assert audit["public_arc_claim_eligibility"] is False
    assert audit["solve_claim_count"] == 0
    assert audit["solve_registry_modified"] is False
    assert audit["claims_ledger_modified"] is False
    assert report["public_claim_eligibility"] is False
    assert report["model_policy_gpu_and_tokenizer_checks"]["all_llm_policy_checks_passed"] is True
    assert report["live_arc_self_discovery_gap_state"]["state"] == "partial_internal_route_only"
    assert report["hardware_gap_state"]["state"] == "unchanged_no_hardware_claim"
    assert report["decentralization_state"]["state"] == "local_host_preserved"


def test_scenario_capstone_6403_schema_edges_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CAPSTONE-6403-FIELD-PRINCIPLES: schema guards fail closed."""

    report = _report()

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    for key in (
        "gate.exp6395.universal_support_claimed",
        "prd.fr11",
        "promotion.arc_route",
        "solve_boundary.arc",
        "public_claim_eligibility",
    ):
        assert key in report["field_principles"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["verifier_is_oracle"] is False
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    bad = deepcopy(report)
    bad["public_claim_eligibility"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "public_claim_eligibility must be false" in mod.validate_report(bad)

    bad = deepcopy(report)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["inference_substrate"] = "live_llm_inference"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_report(bad)

    for field, message in (
        ("active_roadmap_modified", "active_roadmap_modified must be false"),
        ("conductor_modified", "conductor_modified must be false"),
        ("solve_registry_modified", "solve_registry_modified must be false"),
        ("claims_ledger_modified", "claims_ledger_modified must be false"),
    ):
        bad = deepcopy(report)
        bad[field] = True
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert message in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["recomputed_gate_type_finiteness_identity_hash_and_principle_checks"]["by_gate"][
        "exp6395.held_factor_transport_license_ready_score"
    ]["actual"] = {"value": 1.0}
    bad["recomputed_gate_type_finiteness_identity_hash_and_principle_checks"]["by_gate"][
        "exp6395.held_factor_transport_license_ready_score"
    ]["passed"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "gate has nested actual but passed" in mod.validate_report(bad)

    bad = deepcopy(report)
    del bad["field_principles"]["gate.exp6395.universal_support_claimed"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: gate.exp6395.universal_support_claimed" in (
        mod.validate_report(bad)
    )

    bad = deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["field_provenance"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must be a mapping" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["protected_files_unchanged"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "protected files changed" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    assert mod.compare_gate_value({"value": 1.0}, "==", 1.0)["reason"] == "actual_not_bare"
    assert mod.compare_gate_value(float("nan"), "==", 1.0)["reason"] == "actual_not_finite"
    assert mod.compare_gate_value(True, "==", 1.0)["reason"] == "actual_not_bare"
    assert mod.compare_gate_value(1.0, "==", float("nan"))["reason"] == "expected_not_finite"
    assert mod.compare_gate_value(1.0, "==", {"value": 1.0})["reason"] == "expected_not_bare"
    assert mod.compare_gate_value(1.0, "??", 1.0)["reason"] == "unsupported_operator"
    assert mod._receipt_hf_ids([{"hf_id": "model-a"}, {"missing": "model-b"}]) == {"model-a"}
    assert mod._receipt_hf_ids({"model-c": {"ok": True}}) == {"model-c"}
    assert mod._receipt_hf_ids(None) == set()
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._artifact_class(tmp_path, Path("missing.json")) == "absent"
    assert mod._oracle_timing_passed({"exp6401": {"oracle_timing_receipts": []}}) is False

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None, e2e_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_capstone_6403_run_validates_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CAPSTONE-6403: run writes only after validation passes."""

    writes: list[dict[str, object]] = []

    def fake_build_report(
        root: Path,
        *,
        date: str,
        command_receipts: list[dict[str, object]],
        e2e_receipts: list[dict[str, object]],
        before_hashes: dict[str, str | None],
        duration_s: float,
    ) -> dict[str, object]:
        report = {
            "date": date,
            "command_receipts": command_receipts,
            "e2e_receipts": e2e_receipts,
            "before_hashes": before_hashes,
            "duration_s": duration_s,
            "reproducibility_checksum": "",
        }
        report["reproducibility_checksum"] = mod.payload_checksum(report)
        return report

    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    monkeypatch.setattr(mod, "build_report", fake_build_report)
    monkeypatch.setattr(mod, "validate_report", lambda report: [])
    monkeypatch.setattr(mod, "write_report", lambda report, root: writes.append(dict(report)))

    report = mod.run(
        date="20260813",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
        e2e_receipts=[{"command": "e2e", "exit_code": 0}],
    )
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(
            date="20260813",
            root=REPO,
            write=False,
            command_receipts=[],
            e2e_receipts=[],
            duration_s=1.0,
        )

    monkeypatch.setattr(mod, "EXPECTED_ARTIFACTS", {"x": Path("x.json")})
    monkeypatch.setattr(mod, "_artifact_class", lambda root, rel: "weird")
    matrix = mod._terminal_matrix(
        tmp_path,
        {"x": {}, "exp6395": {}},
        {"x": {"present": False, "sha256": None}},
    )
    assert matrix["class_counts"]["weird"] == 1
