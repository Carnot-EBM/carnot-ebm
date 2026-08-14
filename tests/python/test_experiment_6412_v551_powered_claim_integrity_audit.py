"""Tests for Exp6412 V551 powered claim integrity audit.

Spec refs: REQ-REPORT-6412, SCENARIO-REPORT-6412-1,
SCENARIO-REPORT-6412-2, SCENARIO-REPORT-6412-3,
SCENARIO-REPORT-6412-4, SCENARIO-REPORT-6412-5,
SCENARIO-REPORT-6412-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6412_v551_powered_claim_integrity_audit as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _command_receipts() -> list[dict[str, object]]:
    return [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _determination() -> dict[str, object]:
    return {
        "command": ".venv/bin/python scripts/determination_preservation_lint.py",
        "exit_code": 0,
        "stdout": "determination-preservation-lint: OK\n",
        "stderr": "",
    }


def _report(tmp_path: Path, *, write_sidecars: bool = True) -> dict[str, object]:
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    before = mod.protected_hashes(REPO)
    return mod.build_report(
        REPO,
        date="20260814",
        command_receipts=_command_receipts(),
        determination_result=_determination(),
        before_hashes=before,
        duration_s=1.0,
        result_path=result_path,
        write_sidecars=write_sidecars,
    )


def test_req_report_6412_spec_declares_fields_and_scenarios() -> None:
    """REQ-REPORT-6412: OpenSpec owns the Exp6412 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6412") : text.index("REQ-REPORT-6143")]

    for marker in (
        "SCENARIO-REPORT-6412-1",
        "SCENARIO-REPORT-6412-2",
        "SCENARIO-REPORT-6412-3",
        "SCENARIO-REPORT-6412-4",
        "SCENARIO-REPORT-6412-5",
        "SCENARIO-REPORT-6412-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6412_field_matrix_traces_claim_sources(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6412-1 and 2: provenance classes are explicit."""

    report = _report(tmp_path)

    assert mod.validate_report(report) == []
    matrix = report["per_experiment_field_provenance_matrix"]
    exp6408 = matrix["exp6408-powered-write-time-factor-admission-ab"]
    exp6409 = matrix["exp6409-graph-local-multisession-continuous-learning"]

    assert exp6408["runtime.duration_s"]["classification"] == "constant"
    assert exp6408["runtime.duration_s"]["source_lines"]
    assert exp6408["runtime.peak_memory_mb"]["classification"] == "derived"
    assert exp6408["raw_model_bytes_sha256"]["classification"] == "derived"
    assert exp6408["future_exact_success_count"]["classification"] == "constant"
    assert exp6409["runtime.duration_s"]["classification"] == "inherited"
    assert exp6409["future_exact_success_count"]["classification"] == "constant"
    assert exp6409["matched_work.llm_call_count"]["classification"] == "derived"

    model_open = report["model_file_open_evidence"]
    assert model_open["exp6408"]["model_paths_present"] is True
    assert model_open["exp6408"]["model_file_opened_by_model_runtime"] is False
    assert model_open["exp6408"]["classification"] == "absent"

    process = report["model_process_execution_evidence"]
    assert process["exp6408"]["model_process_ran"] is False
    assert process["exp6409"]["model_process_ran"] is False

    token = report["token_generation_evidence"]
    assert token["exp6408"]["generated_token_count_present"] is False
    assert token["exp6409"]["classification"] == "absent"

    raw = report["raw_output_byte_evidence"]
    assert raw["exp6408"]["raw_model_hashes_present"] is True
    assert raw["exp6408"]["raw_bytes_are_model_output"] is False

    gpu = report["pid_bound_gpu_telemetry_evidence"]
    assert gpu["exp6408"]["pid_bound_gpu_samples_present"] is False
    assert gpu["exp6409"]["classification"] == "absent"

    temporal = report["exact_outcome_temporal_evidence"]
    assert temporal["exp6408"]["future_success_counts_source"] == "source_constant_formula"
    assert temporal["exp6409"]["observed_after_admission"] is False


def test_scenario_report_6412_claim_boundary_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6412-3, 5, and 6: powered claims fail closed."""

    report = _report(tmp_path)

    assert report["deterministic_protocol_claim_eligibility"]["eligible"] is False
    assert report["deterministic_replay_claim_eligibility"]["eligible"] is True
    assert report["powered_gguf_claim_eligibility"]["eligible"] is False
    assert report["prospective_csl_claim_eligibility"]["eligible"] is False
    assert report["public_factor_claim_eligibility"]["eligible"] is False
    assert report["fr11_claim_eligibility"]["eligible"] is False
    assert report["v551_claim_boundary_ready_score"] == 1.0
    assert report["powered_false_accept_count"] == 0

    attacks = report["mutation_attack_matrix"]
    assert set(attacks["attacks"]) == set(mod.MUTATION_ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    for row in attacks["attacks"].values():
        assert row["powered_eligible_after_attack"] is False

    for key in (
        "deterministic_protocol_claim_eligibility",
        "deterministic_replay_claim_eligibility",
        "powered_gguf_claim_eligibility",
        "prospective_csl_claim_eligibility",
        "public_factor_claim_eligibility",
        "fr11_claim_eligibility",
        "v551_claim_boundary_ready_score",
    ):
        assert key in report["field_principles"]


def test_scenario_report_6412_additive_sidecars_preserve_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6412-4: sidecars are additive and protected files stay fixed."""

    report = _report(tmp_path)

    corrigendum = report["additive_corrigendum_path_and_hash"]
    ledger = report["additive_claim_ledger_path_entry_and_hash"]
    assert Path(corrigendum["path"]).is_file()
    assert Path(ledger["path"]).is_file()
    assert mod.sha256_file(corrigendum["path"]) == corrigendum["sha256"]
    assert mod.sha256_file(ledger["path"]) == ledger["sha256"]

    ledger_lines = Path(ledger["path"]).read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 1
    assert json.loads(ledger_lines[0]) == ledger["entry"]

    assert report["historical_artifacts_modified"] is False
    assert report["historical_determinations_preserved"] is True
    assert report["protected_files_unchanged"]["ok"] is True
    assert report["protected_files_unchanged"]["changed_paths"] == []
    assert report["preconditions_checked"]["existing_ops_claim_ledger"]["present"] is False


def test_req_report_6412_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-REPORT-6412: validation rejects oracle and powered overclaims."""

    report = _report(tmp_path)
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    bad = copy.deepcopy(report)
    bad["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["powered_false_accept_count"] = 1
    assert "powered false accepts must be zero" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["powered_gguf_claim_eligibility"]["eligible"] = True
    assert "powered GGUF eligibility must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["prospective_csl_claim_eligibility"]["eligible"] = True
    assert "prospective CSL eligibility must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["public_factor_claim_eligibility"]["eligible"] = True
    assert "public factor eligibility must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["fr11_claim_eligibility"]["eligible"] = True
    assert "FR11 eligibility must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"].pop("status")
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"]["status"] = "oracle"
    assert "field_provenance has invalid class" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"].pop("status")
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"].pop("powered_gguf_claim_eligibility")
    assert (
        "missing field_principles entry: powered_gguf_claim_eligibility"
        in mod.validate_report(bad)
    )

    bad = copy.deepcopy(report)
    bad["mutation_attack_matrix"]["all_attacks_fail_closed"] = False
    assert "mutation attacks must fail closed" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["historical_artifacts_modified"] = True
    assert "historical artifacts must not be modified" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["historical_determinations_preserved"] = False
    assert "historical determinations must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["protected_files_unchanged"]["ok"] = False
    assert "protected files changed" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v551_claim_boundary_ready_score"] = 0.0
    assert "ready score mismatch" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "blocked_claim_boundary"
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad.pop("status")
    assert mod.validate_report(bad)[0].startswith("missing required fields:")

    unready = copy.deepcopy(report)
    unready["v551_claim_boundary_ready_score"] = 0.0
    assert mod.status(unready) == "complete_claim_boundary_unready"
    unready["status"] = "complete_claim_boundary_unready"
    assert mod.honest_verdict(unready).startswith("complete_null:")


def test_scenario_report_6412_powered_evidence_requires_all_receipts() -> None:
    """SCENARIO-REPORT-6412-2 and 5: every powered receipt is required."""

    baseline = {
        "model_file_opened": True,
        "model_process_ran": True,
        "tokens_generated": True,
        "raw_output_bytes_stored": True,
        "pid_bound_gpu_samples": True,
        "exact_outcomes_observed_after_admission": True,
        "nonconstant_runtime_duration": True,
    }
    assert mod.powered_evidence_eligible(baseline) is True

    for attack_id in mod.MUTATION_ATTACK_IDS:
        mutated = mod.mutated_powered_evidence(attack_id, baseline)
        assert mod.powered_evidence_eligible(mutated) is False

    with pytest.raises(ValueError, match="unknown_mutation_attack"):
        mod.mutated_powered_evidence("unknown", baseline)


def test_req_report_6412_receipt_loader_and_error_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6412: loader and run error edges are deterministic."""

    not_object = tmp_path / "not_object.json"
    not_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="json_top_level_not_object"):
        mod.read_json_mapping(not_object)

    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(
        json.dumps([{"command": "from-file", "exit_code": 0}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts({}) == [{"command": "from-file", "exit_code": 0}]

    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", tmp_path / "missing.json")
    defaults = mod.read_external_test_receipts({})
    assert defaults[0]["command"] == mod.DEFAULT_TEST_COMMANDS[0]

    monkeypatch.setattr(mod, "build_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "validate_report", lambda report: ["forced error"])
    with pytest.raises(ValueError, match="forced error"):
        mod.run(date="20260814", write=False)


def test_scenario_report_6412_main_writes_to_artifact_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6412-4 and 6: CLI writes additive artifacts safely."""

    out_root = tmp_path / "artifacts"
    out_root.mkdir()
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(out_root))
    monkeypatch.setenv(
        "CARNOT_EXP6412_TEST_RECEIPTS",
        json.dumps(_command_receipts(), sort_keys=True),
    )

    assert mod.main(["--date", "20260814"]) == 0
    captured = json.loads(capsys.readouterr().out)
    written = out_root / mod.RESULT_RELATIVE_PATH.name

    assert captured["path"] == mod.RESULT_RELATIVE_PATH.as_posix()
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_claim_boundary_ready"
    assert payload["v551_claim_boundary_ready_score"] == 1.0
    assert mod.validate_report(payload) == []
