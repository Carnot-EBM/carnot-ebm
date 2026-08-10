"""Tests for Exp6286 V541 evidence eligibility ledger.

Spec refs: REQ-INFRA-6286, SCENARIO-INFRA-6286-1,
SCENARIO-INFRA-6286-2, SCENARIO-INFRA-6286-3,
SCENARIO-INFRA-6286-4, SCENARIO-INFRA-6286-5.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_6286_v541_evidence_eligibility_ledger as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"
GOOD_RAW = "ANSWER: NONE"
GOOD_RAW_HASH = hashlib.sha256(GOOD_RAW.encode("utf-8")).hexdigest()
EMPTY_RAW_HASH = hashlib.sha256(b"").hexdigest()


def _event_row(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "carnot.exp6275.evaluation.row.v1",
        "model_hf_id": "model/a-GGUF",
        "task_id": "task-1",
        "fixture_id": "fixture-1",
        "family": "fixture-family",
        "arm": "one_shot",
        "prompt_hash": "p" * 64,
        "formal_sidecar_hash": "s" * 64,
        "raw_output_hashes": [GOOD_RAW_HASH],
        "parse_success": True,
        "semantic_valid": False,
        "exact_certificate_present": True,
        "abstention": False,
        "parser": "strict",
        "parsed_labels": [],
        "residual_rule_violation_count": 0,
        "residual_rule_violations": [],
        "complete_provenance": True,
    }
    row.update(extra)
    return row


def _raw_row(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "carnot.exp6275.raw_output.v1",
        "model_hf_id": "model/a-GGUF",
        "task_id": "task-1",
        "sample_index": 0,
        "seed": 11,
        "prompt_text": "Solve the task.",
        "prompt_hash": "p" * 64,
        "raw_output": GOOD_RAW,
        "raw_output_hash": GOOD_RAW_HASH,
        "generated_token_count": 2,
        "prompt_token_count": 4,
        "latency_s": 1.0,
        "finish_reason": "stop",
        "timeout": False,
    }
    row.update(extra)
    return row


def _seed_matrix() -> dict[str, object]:
    return {
        "matrix": {
            "model/a-GGUF": [
                {
                    "task_id": "task-1",
                    "sample_index": 0,
                    "seed": 11,
                    "arm_samples": {
                        "one_shot": [11],
                        "self_consistency": [11, 12],
                        "energy_guided_repair": [11],
                    },
                },
                {
                    "task_id": "task-1",
                    "sample_index": 1,
                    "seed": 12,
                    "arm_samples": {
                        "one_shot": [],
                        "self_consistency": [12],
                        "energy_guided_repair": [],
                    },
                },
            ]
        }
    }


def test_req_infra_6286_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6286: OpenSpec records the Exp6286 ledger contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6286") :]

    for token in (
        "REQ-INFRA-6286",
        "SCENARIO-INFRA-6286-1",
        "SCENARIO-INFRA-6286-2",
        "SCENARIO-INFRA-6286-3",
        "SCENARIO-INFRA-6286-4",
        "SCENARIO-INFRA-6286-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6286_raw_validation_conserves_and_quarantines() -> None:
    """SCENARIO-INFRA-6286-2 and SCENARIO-INFRA-6286-3: raw provenance gates rows."""

    missing_output = _raw_row(
        seed=12,
        sample_index=1,
        raw_output="",
        raw_output_hash=EMPTY_RAW_HASH,
    )
    events = [
        _event_row(arm="one_shot"),
        _event_row(arm="self_consistency", raw_output_hashes=[GOOD_RAW_HASH, EMPTY_RAW_HASH]),
        _event_row(arm="energy_guided_repair", formal_sidecar_hash=""),
        _event_row(arm="one_shot", parse_success=None),
    ]

    result = mod.validate_flagship_raw_rows(
        events,
        {"model/a-GGUF": [_raw_row(), missing_output]},
        _seed_matrix(),
    )

    assert result["source_row_count"] == 4
    assert result["eligible_count"] == 1
    assert result["quarantined_count"] == 3
    assert result["eligible_count"] + result["quarantined_count"] == result["source_row_count"]
    reasons = {reason for row in result["quarantine_rows"] for reason in row["missing"]}
    assert "missing_model_output" in reasons
    assert "missing_formal_sidecar_hash" in reasons
    assert "missing_parse_success" in reasons
    assert result["validation_mode"] == "provenance_only_no_scientific_rescoring"


def test_scenario_infra_6286_report_blocks_claim_laundering() -> None:
    """SCENARIO-INFRA-6286-1: valid raw receipts do not reopen flagged artifacts."""

    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert report["flagship_artifact_eligibility"]["artifact_gate_eligible"] is False
    assert report["flagship_artifact_eligibility"]["artifact_level_readiness_closed"] is True
    assert report["flagship_artifact_eligibility"]["raw_rows_reopen_artifact_claim"] is False
    assert report["asp_compiler_eligibility"]["source_module_reusable"] is True
    assert report["typed_backend_eligibility"]["source_module_reusable"] is True
    assert report["dual_cache_treatment_eligibility"]["v542_extension_eligible"] is False
    assert report["mode_jump_treatment_eligibility"]["v542_extension_eligible"] is False
    assert report["arc_router_source_eligibility"]["source_module_reusable"] is True
    assert report["arc_result_eligibility"]["artifact_gate_eligible"] is False
    assert (
        report["eligible_flagship_raw_row_count"] + report["quarantined_flagship_raw_row_count"]
        == report["flagship_raw_manifest_paths_and_hashes"]["event_corpus"]["row_count"]
    )
    assert report["no_claim_laundering_receipt"]["flagged_artifact_promoted"] is False


def test_scenario_infra_6286_validation_catches_hash_drift_and_source_mutation() -> None:
    """SCENARIO-INFRA-6286-4 and SCENARIO-INFRA-6286-5: report validation fails closed."""

    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    assert mod.validate_report(report) == []

    drift = deepcopy(report)
    first_path = next(iter(drift["protected_files_unchanged"]["paths"]))
    drift["protected_files_unchanged"]["paths"][first_path]["after_sha256"] = "sha256:drift"
    drift["protected_files_unchanged"]["paths"][first_path]["unchanged"] = False
    drift["protected_files_unchanged"]["unchanged"] = False
    drift["reproducibility_checksum"] = mod.payload_checksum(drift)
    assert "protected_files_unchanged reports drift" in mod.validate_report(drift)

    mutated = deepcopy(report)
    mutated["source_mutation_count"] = 0.0
    mutated["reproducibility_checksum"] = mod.payload_checksum(mutated)
    assert "source_mutation_count must be bare integer 0" in mod.validate_report(mutated)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad_checksum)

    bad_prefix = deepcopy(report)
    bad_prefix["honest_verdict"] = "clean ledger"
    bad_prefix["reproducibility_checksum"] = mod.payload_checksum(bad_prefix)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad_prefix)


def test_req_infra_6286_write_report_emits_sidecars_under_artifact_root(tmp_path: Path) -> None:
    """REQ-INFRA-6286: report and row manifests are atomic and test-isolated."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    sidecars = report["flagship_raw_row_eligibility_manifest_path_and_hash"]

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report
    for key in ("eligible_manifest", "quarantine_receipt"):
        sidecar_path = artifact_root / Path(sidecars[key]["path"]).name
        assert sidecar_path.exists()
        assert sidecars[key]["sha256"] == mod.sha256_text(sidecar_path.read_text(encoding="utf-8"))


def test_scenario_infra_6286_helper_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6286-3 through SCENARIO-INFRA-6286-5: edge cases fail closed."""

    assert mod._read_json_mapping(tmp_path / "missing.json") == {}
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("\n{}\n[]\n", encoding="utf-8")
    assert mod._read_jsonl(tmp_path / "absent.jsonl") == []
    assert mod._read_jsonl(jsonl) == [{}]
    assert mod._row_path(tmp_path, "") == tmp_path / "missing"
    assert mod._row_path(tmp_path, jsonl) == jsonl
    assert mod._score({"wrapped": {"value": 1}}, "wrapped") == 1

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "V541_ARTIFACTS", {"missing-task": Path("missing.json")})
        assert mod.current_rule_adversarial_results(tmp_path)["missing-task"]["skipped"] == (
            "missing_artifact"
        )

    assert mod._seed_lookup({"matrix": "bad"}) == {}
    bad_seed_matrix = {
        "matrix": {
            "model/a-GGUF": [
                "bad",
                {"task_id": "task", "arm_samples": "bad"},
                {"task_id": "task", "arm_samples": {"one_shot": "bad"}},
                {"task_id": "task", "arm_samples": {"one_shot": ["not-int"]}},
            ],
            "bad-model": "bad",
        }
    }
    assert mod._seed_lookup(bad_seed_matrix) == {("model/a-GGUF", "task", "one_shot"): []}
    assert mod._raw_lookup({"model/a-GGUF": ["bad", {"seed": "bad"}]}) == {}
    assert (
        mod.flagship_raw_manifest_paths_and_hashes(
            tmp_path, {"raw_output_paths_and_hashes": {"model/a-GGUF": "bad"}}
        )["raw_outputs_by_model"]
        == {}
    )

    missing_seed_result = mod.validate_flagship_raw_rows(
        [_event_row()],
        {"model/a-GGUF": []},
        _seed_matrix(),
    )
    assert missing_seed_result["quarantine_rows"][0]["missing"] == ["missing_raw_sample"]

    bad_event = _event_row(
        model_hf_id="",
        task_id="",
        arm="",
        prompt_hash="",
        formal_sidecar_hash="",
        raw_output_hashes=[],
        complete_provenance=False,
        exact_certificate_present=False,
    )
    bad_event.pop("residual_rule_violation_count")
    event_result = mod.validate_flagship_raw_rows([bad_event], {}, {})
    event_reasons = set(event_result["quarantine_rows"][0]["missing"])
    assert {
        "missing_model_hf_id",
        "missing_task_id",
        "missing_arm",
        "missing_prompt_hash",
        "missing_formal_sidecar_hash",
        "missing_raw_output_hashes",
        "missing_complete_provenance",
        "missing_exact_certificate",
        "missing_residual_rule_violation_count",
        "missing_seed",
    }.issubset(event_reasons)

    wrong_raw = _raw_row(
        prompt_text="",
        prompt_hash="bad",
        prompt_token_count="x",
        raw_output="mismatch",
        raw_output_hash="bad",
    )
    wrong_result = mod.validate_flagship_raw_rows(
        [_event_row()],
        {"model/a-GGUF": [wrong_raw]},
        _seed_matrix(),
    )
    wrong_reasons = set(wrong_result["quarantine_rows"][0]["missing"])
    assert {
        "missing_prompt",
        "prompt_hash_mismatch",
        "missing_token_count",
        "raw_hash_not_in_event_row",
        "raw_output_hash_mismatch",
    }.issubset(wrong_reasons)

    no_hash_raw = _raw_row(raw_output_hash="")
    no_hash_result = mod.validate_flagship_raw_rows(
        [_event_row()],
        {"model/a-GGUF": [no_hash_raw]},
        _seed_matrix(),
    )
    assert "missing_raw_output_hash" in no_hash_result["quarantine_rows"][0]["missing"]

    missing_token_raw = _raw_row(seed="11", prompt_hash="")
    missing_token_raw.pop("generated_token_count")
    missing_token_result = mod.validate_flagship_raw_rows(
        [_event_row()],
        {"model/a-GGUF": [missing_token_raw]},
        _seed_matrix(),
    )
    token_reasons = set(missing_token_result["quarantine_rows"][0]["missing"])
    assert {"missing_seed", "missing_prompt_hash", "missing_token_count"}.issubset(token_reasons)

    blocked = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 1}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")

    invalid = {"status": "complete"}
    errors = mod.validate_report(invalid)
    assert "missing required field: v541_capstone_path_hash_and_terminal_class" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "raw row counts must be integers" in errors
    assert "wrong inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "claim laundering receipt promoted a flagged artifact" in errors
    assert "reproducibility_checksum missing" in errors

    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    mismatched = deepcopy(report)
    mismatched["eligible_flagship_raw_row_count"] += 1
    mismatched["reproducibility_checksum"] = mod.payload_checksum(mismatched)
    assert "eligible and quarantined raw rows do not sum to source row count" in (
        mod.validate_report(mismatched)
    )

    bad_claim = deepcopy(report)
    bad_claim["no_claim_laundering_receipt"]["flagged_artifact_promoted"] = True
    bad_claim["flagship_artifact_eligibility"]["artifact_gate_eligible"] = True
    bad_claim["arc_result_eligibility"]["artifact_gate_eligible"] = True
    bad_claim["reproducibility_checksum"] = mod.payload_checksum(bad_claim)
    claim_errors = mod.validate_report(bad_claim)
    assert "claim laundering receipt promoted a flagged artifact" in claim_errors
    assert "flagship artifact gate must remain closed" in claim_errors
    assert "arc result gate must remain closed" in claim_errors

    with pytest.raises(ValueError, match="invalid Exp6286 report"):
        mod.write_report(invalid, REPO, env={ARTIFACT_ROOT_ENV: str(tmp_path)})
