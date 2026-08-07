"""Tests for Exp6197 terminal-artifact contract deliverable.

Spec refs: REQ-INFRA-6197, SCENARIO-INFRA-6197-3,
SCENARIO-INFRA-6197-4, SCENARIO-INFRA-6197-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6197_v537_terminal_artifact_contract as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _fake_runner(argv: tuple[str, ...], _root: Path) -> JsonDict:
    command = " ".join(argv)
    return {
        "command": command,
        "exit_code": 2 if command == mod.FULL_SUITE_COMMAND else 0,
        "classification": (
            "unrelated_preexisting_nonzero" if command == mod.FULL_SUITE_COMMAND else "passed"
        ),
    }


def _build() -> JsonDict:
    return mod.build_report(REPO, date="20260807", command_runner=_fake_runner, duration_s=1.25)


def test_scenario_infra_6197_spec_declares_contract() -> None:
    """REQ-INFRA-6197: OpenSpec names the classifier and deliverable."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6197") :]

    for marker in (
        "REQ-INFRA-6197",
        "SCENARIO-INFRA-6197-1",
        "SCENARIO-INFRA-6197-2",
        "SCENARIO-INFRA-6197-3",
        "SCENARIO-INFRA-6197-4",
        "SCENARIO-INFRA-6197-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/terminal_artifacts.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6197_build_report_classifies_fixtures_and_receipts() -> None:
    """SCENARIO-INFRA-6197-4: Exp6183 and Exp6196 remain nonterminal."""

    report = _build()

    assert report["status"] == "complete_ready"
    assert report["conductor_receipt_override_count"] == 0
    assert type(report["conductor_receipt_override_count"]) is int
    assert report["protected_artifact_mutation_count"] == 0
    assert type(report["protected_artifact_mutation_count"]) is int
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert report["honest_verdict"].startswith("complete:")

    assert report["exp6183_classification"]["terminal"] is False
    assert report["exp6183_classification"]["classification"] == "running_bootstrap"
    assert report["exp6183_classification"]["receipt_override_attempted"] is True
    assert report["exp6196_classification"]["terminal"] is False
    assert report["exp6196_classification"]["classification"] == "running_bootstrap"
    assert report["exp6196_classification"]["receipt_overrode"] is False

    classes = {
        row["fixture_id"]: row["classification"]
        for row in report["valid_fixture_classifications"]
    }
    assert classes["exp482_complete"] == "complete"
    assert classes["exp6194_ready"] == "ready"
    assert classes["exp6195_positive"] == "positive"
    assert classes["exp6193_gated"] == "skipped"
    assert classes["exp6175_retired"] == "retired"
    assert classes["exp6187_flagged"] == "flagged"
    assert classes["exp411_blocked"] == "blocked"
    assert classes["exp1239_running"] == "running"
    assert classes["missing_declared_path"] == "missing"
    assert classes["malformed_pcib"] == "malformed"
    assert classes["synthetic_running"] == "running"
    assert classes["synthetic_bootstrap_only"] == "bootstrap_only"


def test_scenario_infra_6197_required_fields_principles_and_checksum() -> None:
    """SCENARIO-INFRA-6197-5: required fields are principled and stable."""

    report = _build()

    assert mod._pytest_failures(
        "FAILED tests/python/test_x.py::test_a - AssertionError\nok"
    ) == ["tests/python/test_x.py::test_a"]
    assert mod.validate_report(report) == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(report)
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_principles"][field]
        provenance = report["field_provenance"][field]
        assert provenance["principle"] == report["field_principles"][field]
        assert provenance["source"]

    cross_product = report["status_verdict_cross_product"]
    assert cross_product["summary"]["terminal_pair_count"] > 0
    assert cross_product["summary"]["nonterminal_pair_count"] > 0
    assert cross_product["summary"]["contradictory_pair_count"] > 0
    assert report["classifier_module_and_hash"]["imports_conductor"] is False
    assert report["classifier_module_and_hash"]["sha256"].startswith("sha256:")
    assert report["full_suite_command_and_classified_exit_code"]["exit_code"] == 2
    assert (
        report["full_suite_command_and_classified_exit_code"]["classification"]
        == "unrelated_preexisting_nonzero"
    )


def test_scenario_infra_6197_write_contract_uses_temp_output_root(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6197-5: writer atomically emits one JSON artifact."""

    report = mod.write_contract(
        REPO,
        date="20260807",
        command_runner=_fake_runner,
        duration_s=1.25,
        env={ARTIFACT_ROOT_ENV: str(tmp_path)},
    )
    target = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert target.exists()
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == report
    assert mod.validate_report(report) == []
    assert not hasattr(mod, "write_bootstrap")


def test_req_infra_6197_validate_report_rejects_bad_outputs() -> None:
    """REQ-INFRA-6197: malformed Exp6197 artifacts fail validation."""

    report = _build()

    missing = deepcopy(report)
    missing.pop("status")
    assert "missing:status" in mod.validate_report(missing)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_substrate = deepcopy(report)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    assert "inference_substrate" in mod.validate_report(bad_substrate)

    bad_oracle = deepcopy(report)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    assert "verifier_is_oracle" in mod.validate_report(bad_oracle)

    bad_zero = deepcopy(report)
    bad_zero["conductor_receipt_override_count"] = "0"
    bad_zero["reproducibility_checksum"] = mod.payload_checksum(bad_zero)
    assert "conductor_receipt_override_count" in mod.validate_report(bad_zero)

    bad_focused = deepcopy(report)
    bad_focused["focused_test_exit_codes"][mod.FOCUSED_TEST_COMMANDS[0]] = 1
    bad_focused["reproducibility_checksum"] = mod.payload_checksum(bad_focused)
    assert "focused_test_exit_codes" in mod.validate_report(bad_focused)

    bad_provenance_type = deepcopy(report)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.payload_checksum(bad_provenance_type)
    assert "field_provenance" in mod.validate_report(bad_provenance_type)

    bad_principles_type = deepcopy(report)
    bad_principles_type["field_principles"] = []
    bad_principles_type["reproducibility_checksum"] = mod.payload_checksum(bad_principles_type)
    assert "field_principles" in mod.validate_report(bad_principles_type)

    bad_principle = deepcopy(report)
    bad_principle["field_principles"]["status"] = ""
    bad_principle["field_provenance"]["status"]["principle"] = ""
    bad_principle["reproducibility_checksum"] = mod.payload_checksum(bad_principle)
    assert "field_principles:status" in mod.validate_report(bad_principle)

    bad_provenance_principle = deepcopy(report)
    bad_provenance_principle["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance_principle["reproducibility_checksum"] = mod.payload_checksum(
        bad_provenance_principle
    )
    assert "field_provenance:status" in mod.validate_report(bad_provenance_principle)

    bad_exp = deepcopy(report)
    bad_exp["exp6183_classification"]["terminal"] = True
    bad_exp["reproducibility_checksum"] = mod.payload_checksum(bad_exp)
    assert "exp6183_classification" in mod.validate_report(bad_exp)

    bad_exp_class = deepcopy(report)
    bad_exp_class["exp6196_classification"]["classification"] = "missing"
    bad_exp_class["reproducibility_checksum"] = mod.payload_checksum(bad_exp_class)
    assert "exp6196_classification" in mod.validate_report(bad_exp_class)

    bad_fixtures = deepcopy(report)
    bad_fixtures["valid_fixture_classifications"][0]["matches_expected"] = False
    bad_fixtures["reproducibility_checksum"] = mod.payload_checksum(bad_fixtures)
    assert "valid_fixture_classifications" in mod.validate_report(bad_fixtures)

    bad_classifier = deepcopy(report)
    bad_classifier["classifier_module_and_hash"]["imports_conductor"] = True
    bad_classifier["reproducibility_checksum"] = mod.payload_checksum(bad_classifier)
    assert "classifier_module_and_hash" in mod.validate_report(bad_classifier)

    monkey_report = deepcopy(report)
    monkey_report["honest_verdict"] = "ready but no prefix"
    monkey_report["reproducibility_checksum"] = mod.payload_checksum(monkey_report)
    assert "honest_verdict" in mod.validate_report(monkey_report)


def test_req_infra_6197_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6197: invalid reports are not written."""

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6197 terminal contract"):
        mod.write_contract(
            REPO,
            date="20260807",
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(tmp_path)},
        )
    assert not (tmp_path / mod.RESULT_RELATIVE_PATH.name).exists()
