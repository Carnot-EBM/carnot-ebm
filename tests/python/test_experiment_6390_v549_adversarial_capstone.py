"""Tests for Exp6390 V549 adversarial capstone.

Spec refs: REQ-CAPSTONE-6390, SCENARIO-CAPSTONE-6390,
SCENARIO-CAPSTONE-6390-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6390_v549_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _receipt(command: str = "focused") -> list[dict[str, object]]:
    return [{"command": command, "exit_code": 0}]


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260813",
        command_receipts=_receipt(),
        before_hashes=mod.protected_hashes(REPO),
        duration_s=1.0,
    )


def test_req_capstone_6390_spec_declares_schema_and_principles() -> None:
    """REQ-CAPSTONE-6390: OpenSpec owns the V549 capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6390") :]

    for token in (
        "SCENARIO-CAPSTONE-6390",
        "SCENARIO-CAPSTONE-6390-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "public_claim_eligibility=false",
        "Exp6382 absence SHALL stay missing",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_capstone_6390_preserves_terminal_classes() -> None:
    """SCENARIO-CAPSTONE-6390: missing, blocked, null, and flagged stay visible."""

    report = _report()
    classes = report["expected_task_ids_and_terminal_classes"]

    assert mod.validate_report(report) == []
    assert classes["classification_before_semantic_reads"] is True
    assert classes["expected_upstream_task_ids"] == list(mod.UPSTREAM_TASK_IDS)
    assert classes["class_counts"]["missing"] == 1
    assert classes["class_counts"]["blocked"] == 3
    assert classes["class_counts"]["null"] == 1
    assert classes["class_counts"]["flagged"] == 2
    assert classes["by_task"]["exp6382-chronological-verified-factor-self-learning"][
        "terminal_class"
    ] == "missing"
    assert classes["by_task"]["exp6380-three-family-canonical-factor-transport-canary"][
        "terminal_class"
    ] == "null"
    assert classes["by_task"]["exp6385-live-factor-learning-and-rollback-safety-audit"][
        "terminal_class"
    ] == "flagged"

    adversarial = report["original_and_live_adversarial_verdicts"]
    assert adversarial["exp6377-v549-terminal-handoff-and-queue-preflight"][
        "stamped_flagged_adversarial"
    ] is True
    assert adversarial["exp6385-live-factor-learning-and-rollback-safety-audit"][
        "live_has_critical"
    ] is True
    assert report["missing_blocked_null_flagged_and_retired_evidence"]["proposal_only"][
        "v548_ids_preserved"
    ] is True


def test_scenario_capstone_6390_recomputes_readiness_and_gates() -> None:
    """SCENARIO-CAPSTONE-6390: gates and readiness come from primary fields."""

    report = _report()
    gates = report["structured_gate_recomputation"]
    readiness = report["readiness_field_recomputation"]
    retirement = report["prior_failure_and_retirement_recomputation"]

    assert readiness["exp6379-canonical-factor-edit-transport-contract"]["recomputed"] == 1.0
    assert readiness["exp6380-three-family-canonical-factor-transport-canary"][
        "recomputed"
    ] == 0.0
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in readiness[
        "exp6380-three-family-canonical-factor-transport-canary"
    ]["blocking_reasons"]
    assert readiness["exp6383-dependency-guided-factor-rollback-stress"][
        "matches_artifact"
    ] is True
    assert readiness["exp6385-live-factor-learning-and-rollback-safety-audit"][
        "clean_evidence"
    ] is False

    assert gates["by_task"]["exp6381-verified-frontier-live-factor-proposal-ab"][
        "all_gates_passed"
    ] is False
    assert gates["by_task"]["exp6384-default-off-certified-factor-consumer-ab"][
        "gate_rows"
    ][0]["reason"] == "upstream_artifact_missing"
    shadow_rows = gates["by_task"]["exp6389-arc-default-off-active-goal-shadow"][
        "gate_rows"
    ]
    assert shadow_rows[1]["actual_type"] == "dict"
    assert shadow_rows[1]["passed"] is False
    assert shadow_rows[1]["numeric_payload_hint"] == 0.75

    assert retirement["exp6380_repeated_all_invalid_verdict"] is False
    assert retirement["exclusion_manifest_update_required"] is False


def test_scenario_capstone_6390_boundaries_decisions_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CAPSTONE-6390-FIELD-PRINCIPLES: boundaries are explicit."""

    report = _report()

    assert report["public_claim_eligibility"] is False
    assert report["verifier_is_oracle"] is False
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["documentation_reconciliation_receipts"]["ops_docs_updated"] is False
    assert report["documentation_reconciliation_receipts"]["deferred_by_stop_rule"] is True
    assert report["factor_transport_verdict"]["decision"] == "null"
    assert report["continuous_self_learning_verdict"]["decision"] == "missing"
    assert report["dependency_rollback_verdict"]["decision"] == "advanced"
    assert report["factor_safety_verdict"]["clean_public_safety_evidence"] is False
    assert report["arc_registry_and_no_solve_audit"]["arc_solve_claim_count"] == 0
    assert report["arc_registry_and_no_solve_audit"]["registry_write_count"] == 0
    assert report["arc_live_shadow_verdict"]["decision"] == "blocked"
    assert report["three_prd_gap_decisions"]["prospective_fr11_with_rollback_and_consumer"][
        "decision"
    ] == "blocked"
    assert report["branch_promotion_retirement_and_deferral_decisions"]["arc_live_shadow"][
        "branch_decision"
    ] == "defer"
    assert report["protected_files_unchanged"]["ok"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    bad = deepcopy(report)
    bad["public_claim_eligibility"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "public_claim_eligibility must be false" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["expected_task_ids_and_terminal_classes"]["by_task"][
        "exp6382-chronological-verified-factor-self-learning"
    ]["terminal_class"] = "clean"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6382 missing state must be preserved" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

    bad = deepcopy(report)
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_capstone_6390_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6390: malformed inputs and bad comparisons fail closed."""

    missing = tmp_path / "missing.json"
    assert mod.read_json_mapping(missing)[1]["error"] == "missing"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"].startswith("json_error:")

    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "json_not_mapping"
    assert mod._load_yaml_mapping(tmp_path / "missing.yaml") == {}

    assert mod.compare_gate_value({"pooled_unrounded": 0.75}, ">", 0.0)["passed"] is False
    assert mod.compare_gate_value(float("nan"), "==", 1.0)["reason"] == "non_finite_actual"
    assert mod.compare_gate_value(1.0, "==", "x")["reason"] == "expected_not_bare_numeric"
    assert mod.compare_gate_value(1.0, "==", 1.0)["passed"] is True
    assert mod.compare_gate_value(2.0, ">", 1.0)["passed"] is True
    assert mod.compare_gate_value(2.0, "<=", 1.0)["passed"] is False
    assert mod.compare_gate_value(2.0, "??", 1.0)["reason"] == "unsupported_operator"

    assert mod._terminal_class({}, {"error": "bad"}) == "malformed"
    assert mod._terminal_class({"honest_verdict": "blocked_gate_check_failed"}, {"error": None}) == "blocked"
    assert mod._terminal_class({"status": "complete_generic"}, {"error": None}) == "positive"
    assert mod._terminal_class({"status": "not_terminal"}, {"error": None}) == "unknown"
    assert mod._referenced_repo_files(REPO, "not-a-path") == set()
    assert mod._all_recursive({"x": False}, "x", True) is False

    gate_root = tmp_path / "gate-root"
    gate_root.mkdir()
    (gate_root / mod.ACTIVE_ROADMAP_RELATIVE_PATH).write_text(
        "tasks:\n  - not-a-mapping\n", encoding="utf-8"
    )
    assert mod._roadmap_gates(gate_root) == {}

    receipt = tmp_path / "receipts.json"
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{}", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text('[{"command": "ok", "exit_code": 0}, "skip"]', encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == [{"command": "ok", "exit_code": 0}]

    report = _report()
    payloads, metas, _summaries = mod._load_upstreams(REPO)
    terminal = report["expected_task_ids_and_terminal_classes"]
    payloads["exp6379-canonical-factor-edit-transport-contract"] = dict(
        payloads["exp6379-canonical-factor-edit-transport-contract"],
        autotokenizer_usage_count=1,
    )
    assert mod._score_row(
        "exp6379-canonical-factor-edit-transport-contract", payloads, metas, terminal
    )["blocking_reasons"] == ["conjunctive_readiness_gate_failed"]

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

    bad = deepcopy(report)
    bad["field_provenance"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must be a mapping" in mod.validate_report(bad)

    bad = deepcopy(report)
    bad["expected_task_ids_and_terminal_classes"]["by_task"][
        "exp6385-live-factor-learning-and-rollback-safety-audit"
    ]["terminal_class"] = "clean"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6385 flagged state must be preserved" in mod.validate_report(bad)

    report["protected_files_unchanged"]["ok"] = False
    report["reproducibility_checksum"] = mod.payload_checksum(report)
    assert "protected files changed" in mod.validate_report(report)


def test_req_capstone_6390_run_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-6390: run validates and writes only after a clean report."""

    writes: list[dict[str, object]] = []

    def fake_build_report(
        root: Path,
        *,
        date: str,
        command_receipts: list[dict[str, object]],
        before_hashes: dict[str, str],
        duration_s: float,
    ) -> dict[str, object]:
        report = {
            "date": date,
            "command_receipts": command_receipts,
            "before_hashes": before_hashes,
            "duration_s": duration_s,
            "reproducibility_checksum": "",
        }
        report["reproducibility_checksum"] = mod.payload_checksum(report)
        return report

    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    monkeypatch.setattr(mod, "read_external_test_receipts", lambda: [{"command": "external"}])
    monkeypatch.setattr(mod, "build_report", fake_build_report)
    monkeypatch.setattr(mod, "validate_report", lambda report: [])
    monkeypatch.setattr(mod, "write_report", lambda report, root: writes.append(dict(report)))

    report = mod.run(date="20260813", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert report["duration_s"] >= 0.0
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(
            date="20260813",
            root=REPO,
            write=False,
            command_receipts=[{"command": "manual"}],
            duration_s=1.0,
        )
