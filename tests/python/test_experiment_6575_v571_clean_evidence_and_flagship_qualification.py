"""Tests for the fresh V571 flagship qualification root.

Spec refs: REQ-REPORT-6575, REQ-REPORT-6575-FRESH,
REQ-REPORT-6575-METADATA, REQ-REPORT-6575-RUNTIME,
REQ-REPORT-6575-RECEIPTS, REQ-REPORT-6575-RESIDENCY,
REQ-REPORT-6575-METHOD, REQ-REPORT-6575-LINKS,
REQ-REPORT-6575-ATTACKS, REQ-REPORT-6575-DURATION,
REQ-REPORT-6575-REDUCER, REQ-REPORT-6575-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6575_v571_clean_evidence_and_flagship_qualification as exp


def _context_rows() -> list[dict]:
    return [
        exp.hash_row(
            {
                "row_type": "v570_ineligible_context",
                "experiment_id": experiment_id,
                "path": f"results/experiment_{experiment_id}_fixture.json",
                "artifact_sha256": f"sha256:{str(index + 1) * 64}",
                "stored_flagged_adversarial": True,
                "structural_findings": ["DURATION_TOO_SHORT"],
                "eligible_for_v571_reducer": False,
                "ineligibility_reason": "stored structural flag",
                "passed": True,
            }
        )
        for index, experiment_id in enumerate((6571, 6572, 6573))
    ]


def _metadata_rows() -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        digest = str(index + 4) * 64
        rows.append(
            exp.hash_row(
                {
                    "row_type": "fresh_metadata_positive",
                    "repository_id": spec["repository_id"],
                    "sequence_index": index,
                    "fresh": True,
                    "selected_blob_path": f"/cache/blobs/{digest}",
                    "trusted_sha256": f"sha256:{digest}",
                    "fresh_full_content_sha256": f"sha256:{digest}",
                    "admitted": True,
                    "content_metadata": {
                        "architecture": spec["expected_architecture"],
                        "quantization": "Q4_K_M",
                        "tensor_count": 100,
                        "is_language_model": True,
                        "tokenizer_metadata": {
                            "token_count": 256_000,
                            "chat_template_present": True,
                        },
                        "bounded_read_receipt": {"tensor_payload_bytes_read": 0},
                    },
                    "provenance": {
                        "valid": True,
                        "repository_id": spec["repository_id"],
                        "revision": f"revision-{index}",
                        "snapshot_filename": f"model-{index}.gguf",
                    },
                    "passed": True,
                }
            )
        )
    return rows


def _negative_rows() -> list[dict]:
    return [
        exp.hash_row(
            {
                "row_type": "fresh_metadata_negative",
                "unit_id": fixture_id,
                "fresh": True,
                "observed_admitted": False,
                "passed": True,
            }
        )
        for fixture_id in exp.REQUIRED_NEGATIVE_FIXTURES
    ]


def _runtime_receipts() -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        output = f"fresh lighthouse qualification output {index}"
        rows.append(
            exp.hash_row(
                {
                    "row_type": "fresh_family_runtime",
                    "repository_id": spec["repository_id"],
                    "sequence_index": index,
                    "fresh": True,
                    "passed": True,
                    "process": {
                        "pid": 8000 + index,
                        "raw_output": output,
                        "raw_output_sha256": exp.sha256_text(output),
                        "output_token_count": 6,
                        "stop_reason": "stop",
                        "exit_code": 0,
                        "normal_shutdown": True,
                        "stderr_sha256": "sha256:" + "a" * 64,
                    },
                    "gpu_samples": [{"stage": "during"}, {"stage": "during"}],
                    "stage_rows": [{"passed": True}],
                    "unload": {"recovery_complete": True},
                }
            )
        )
    return rows


def _unload_rows() -> list[dict]:
    return [
        exp.hash_row(
            {
                "row_type": "fresh_unload_recovery",
                "repository_id": spec["repository_id"],
                "sequence_index": index,
                "fresh": True,
                "normal_shutdown": True,
                "worker_absent_from_proc": True,
                "worker_absent_from_nvidia_smi": True,
                "port_closed": True,
                "no_task_worker_remains": True,
                "recovery_complete": True,
                "signals_sent_to_unrelated_pids": [],
                "passed": True,
            }
        )
        for index, spec in enumerate(exp.MODEL_SPECS)
    ]


def _preconditions(*, ready: bool = True) -> dict:
    return {
        "all_required_preconditions_available": ready,
        "checks": {"cuda_llama_cpp": ready, "cache": ready},
        "failed_preconditions": [] if ready else ["cuda_llama_cpp"],
        "expected_inference_substrate": exp.INFERENCE_SUBSTRATE,
        "model_load_order": list(exp.MANDATED_HF_IDS),
    }


def _protected(*, unchanged: bool = True) -> dict:
    return {
        "all_unchanged": unchanged,
        "research_roadmap_yaml_unchanged": unchanged,
        "research_conductor_py_unchanged": unchanged,
        "rows": [],
    }


def _structural(*, passed: bool = True) -> dict:
    return exp.hash_row(
        {
            "row_type": "live_structural_verification",
            "fresh": True,
            "duration_floor_s": exp.LIVE_DURATION_FLOOR_S,
            "flag_count": 0 if passed else 1,
            "critical_flag_count": 0 if passed else 1,
            "findings": [] if passed else ["DURATION_TOO_SHORT"],
            "passed": passed,
        }
    )


def _tests_run() -> list[dict]:
    return [{"command": "focused Exp6575 tests", "exit_code": 0, "duration_s": 1.0}]


def _assemble(
    *,
    runtime_receipts: list[dict] | None = None,
    method_rows: list[dict] | None = None,
    duration_s: float = 75.0,
    preconditions: dict | None = None,
    protected: dict | None = None,
    structural: dict | None = None,
) -> dict:
    return exp.assemble_artifact(
        context_rows=_context_rows(),
        metadata_rows=_metadata_rows(),
        negative_rows=_negative_rows(),
        runtime_receipts=_runtime_receipts() if runtime_receipts is None else runtime_receipts,
        unload_rows=_unload_rows(),
        method_rows=exp.build_method_replay_rows() if method_rows is None else method_rows,
        attack_rows=exp.build_attack_rows(),
        preconditions=_preconditions() if preconditions is None else preconditions,
        protected=_protected() if protected is None else protected,
        structural_verification=_structural() if structural is None else structural,
        duration_s=duration_s,
        tests_run=_tests_run(),
        run_date="20260824",
    )


# REQ-REPORT-6575: the complete OpenSpec contract must exist before implementation.
def test_spec_declares_every_v571_requirement_and_scenario() -> None:
    text = (exp.REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6575") :]
    for anchor in (
        "REQ-REPORT-6575-FRESH",
        "REQ-REPORT-6575-METADATA",
        "REQ-REPORT-6575-RUNTIME",
        "REQ-REPORT-6575-RECEIPTS",
        "REQ-REPORT-6575-RESIDENCY",
        "REQ-REPORT-6575-METHOD",
        "REQ-REPORT-6575-LINKS",
        "REQ-REPORT-6575-ATTACKS",
        "REQ-REPORT-6575-DURATION",
        "REQ-REPORT-6575-REDUCER",
        "REQ-REPORT-6575-ATOMIC",
        "SCENARIO-REPORT-6575-FLAGGED",
        "SCENARIO-REPORT-6575-METADATA",
        "SCENARIO-REPORT-6575-RUNTIME",
        "SCENARIO-REPORT-6575-METHOD",
        "SCENARIO-REPORT-6575-LINKS",
        "SCENARIO-REPORT-6575-ATTACKS",
        "SCENARIO-REPORT-6575-ATOMIC",
    ):
        assert anchor in section


# REQ-REPORT-6575-FRESH / SCENARIO-REPORT-6575-FLAGGED:
# stored V570 flags stay visible but are never eligible inputs.
def test_v570_context_builder_marks_all_three_artifacts_ineligible(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    for relative_path in exp.V570_CONTEXT_PATHS:
        (tmp_path / relative_path).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "duration_s": 52.0,
                    "flagged_adversarial": True,
                    "v570_evidence_contract_ready_score": 1.0,
                }
            ),
            encoding="utf-8",
        )

    rows = exp.build_v570_context_rows(
        tmp_path,
        verifier=lambda path: {
            "artifact": str(path),
            "flag_count": 1,
            "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )

    assert [row["experiment_id"] for row in rows] == [6571, 6572, 6573]
    assert all(row["eligible_for_v571_reducer"] is False for row in rows)
    assert all("v570_evidence_contract_ready_score" not in row for row in rows)
    assert all(row["structural_findings"] == ["DURATION_TOO_SHORT"] for row in rows)


# REQ-REPORT-6575-METADATA / SCENARIO-REPORT-6575-METADATA:
# fresh cache resolution, bounded inspection, and full content hashes are independent of Exp6572.
def test_fresh_metadata_builder_binds_all_families(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {hf_id: f"/cache/{index + 4:064x}" for index, hf_id in enumerate(exp.MANDATED_HF_IDS)}
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda hf_id: paths[hf_id])
    monkeypatch.setattr(exp.Path, "resolve", lambda self: self)
    monkeypatch.setattr(exp, "sha256_file", lambda path: f"sha256:{Path(path).name}")

    def admission(path: Path, **kwargs: object) -> dict:
        hf_id = str(kwargs["repository_id"])
        spec = next(row for row in exp.MODEL_SPECS if row["repository_id"] == hf_id)
        return {
            "admitted": True,
            "content_metadata": {
                "architecture": spec["expected_architecture"],
                "quantization": "Q4_K_M",
                "tensor_count": 10,
                "is_language_model": True,
                "tokenizer_metadata": {"token_count": 100, "chat_template_present": True},
                "bounded_read_receipt": {"tensor_payload_bytes_read": 0},
            },
            "provenance": {
                "valid": True,
                "repository_id": hf_id,
                "revision": "revision",
                "snapshot_filename": "model.gguf",
            },
            "rejection_reasons": [],
        }

    monkeypatch.setattr(exp, "build_gguf_admission_record", admission)
    rows = exp.build_fresh_metadata_rows()

    assert [row["repository_id"] for row in rows] == list(exp.MANDATED_HF_IDS)
    assert all(row["passed"] for row in rows)
    assert all(row["fresh_full_content_sha256"] == row["trusted_sha256"] for row in rows)
    assert all(row["upstream_v570_readiness_imported"] is False for row in rows)


# REQ-REPORT-6575-METHOD / SCENARIO-REPORT-6575-METHOD:
# the selected Exp6574 fixtures are rebuilt and replayed through the shipped reducer.
def test_method_replay_hashes_fresh_safe_and_unsafe_results() -> None:
    rows = exp.build_method_replay_rows()

    assert [row["fixture_id"] for row in rows] == list(exp.REQUIRED_METHOD_FIXTURES)
    assert [row["action"] for row in rows] == [
        "release",
        "release",
        "abstain",
        "abstain",
        "abstain",
    ]
    assert all(row["passed"] and row["fresh"] for row in rows)
    assert all(row["fixture_sha256"].startswith("sha256:") for row in rows)
    assert all(row["result_sha256"].startswith("sha256:") for row in rows)
    assert all(row["reducer"] == exp.METHOD_REDUCER_NAME for row in rows)


# REQ-REPORT-6575-REDUCER / SCENARIO-REPORT-6575-ATOMIC:
# all fresh row classes are needed for a ready terminal artifact.
def test_ready_artifact_recomputes_both_scores_only_from_fresh_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp, "runtime_receipt_passes", lambda _row, _metadata: True)
    artifact = _assemble()

    assert artifact["status"] == "complete_v571_flagship_qualification_ready"
    assert artifact["verdict_class"] is None
    assert artifact["v571_flagship_evidence_ready_score"] == 1.0
    assert artifact["joint_sufficiency_method_replay_ready_score"] == 1.0
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is True
    assert exp.validate_artifact(artifact) == []


# REQ-REPORT-6575-RUNTIME / REQ-REPORT-6575-DURATION:
# one failed family or a short structural run cannot hide under an aggregate.
def test_failed_family_and_short_duration_fail_closed() -> None:
    runtime_rows = _runtime_receipts()
    runtime_rows[1]["passed"] = False
    runtime_rows[1] = exp.hash_row(runtime_rows[1])
    partial = _assemble(runtime_receipts=runtime_rows)
    assert partial["v571_flagship_evidence_ready_score"] == 0.0
    assert partial["verdict_class"] == "partial"
    assert exp.MANDATED_HF_IDS[1] in partial["gate_check_summary"]["failed_families"]

    short = _assemble(duration_s=59.0, structural=_structural(passed=False))
    assert short["v571_flagship_evidence_ready_score"] == 0.0
    assert short["verdict_class"] == "disqualified"
    assert short["repeat_retirement_rule_activated"] is True


# REQ-REPORT-6575-METHOD: missing fixture coverage closes only the method score directly.
def test_missing_method_fixture_closes_method_and_flagship_scores() -> None:
    artifact = _assemble(method_rows=exp.build_method_replay_rows()[:-1])

    assert artifact["joint_sufficiency_method_replay_ready_score"] == 0.0
    assert artifact["v571_flagship_evidence_ready_score"] == 0.0
    assert artifact["verdict_class"] == "partial"


# REQ-REPORT-6575-ATTACKS / SCENARIO-REPORT-6575-ATTACKS:
# every preregistered runtime or laundering mutation is rejected.
def test_attack_matrix_covers_every_required_failure_mode() -> None:
    rows = exp.build_attack_rows()

    assert {row["attack_id"] for row in rows} == set(exp.REQUIRED_ATTACKS)
    assert all(row["passed"] for row in rows)
    assert all(row["observed_ready_score"] == 0.0 for row in rows)


# REQ-REPORT-6575-LINKS / SCENARIO-REPORT-6575-LINKS:
# checksum, row hashes, source links, and forbidden V570 fields are validated.
def test_validator_detects_link_checksum_runtime_and_laundering_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp, "runtime_receipt_passes", lambda _row, _metadata: True)
    artifact = _assemble()
    artifact["reproducibility_checksum"] = "sha256:bad"
    artifact["evidence_link_rows"][0]["source_row_hashes"] = ["sha256:missing"]
    artifact["v570_ineligible_context_rows"][0]["v570_evidence_contract_ready_score"] = 1.0
    monkeypatch.setattr(exp, "runtime_receipt_passes", lambda _row, _metadata: False)

    errors = exp.validate_artifact(artifact)

    assert "reproducibility_checksum_mismatch" in errors
    assert "evidence_link_source_missing" in errors
    assert "v570_readiness_field_laundering" in errors
    assert "runtime_receipt_recomputation_failed" in errors


# REQ-REPORT-6575-ATOMIC / SCENARIO-REPORT-6575-ATOMIC:
# one same-directory replacement is readable and the validation CLI fails closed.
def test_atomic_write_and_validation_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(exp, "runtime_receipt_passes", lambda _row, _metadata: True)
    artifact = _assemble()
    path = tmp_path / "artifact.json"
    exp.atomic_write_json(path, artifact)

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(exp, "RESULT_RELATIVE_PATH", path.relative_to(tmp_path))
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    assert exp.main(["--validate"]) == 0
    assert "validated" in capsys.readouterr().out
    artifact["verdict_class"] = "positive"
    exp.atomic_write_json(path, artifact)
    assert exp.main(["--validate"]) == 1


# REQ-REPORT-6575-ATOMIC: canonical row and artifact hashes detect mutation.
def test_hash_helpers_are_stable_and_self_fields_are_excluded() -> None:
    assert exp.sha256_json({"a": 1, "b": 2}) == exp.sha256_json({"b": 2, "a": 1})
    row = exp.hash_row({"value": 1})
    assert exp.hash_row(row) == row
    artifact = {"value": 1}
    checksum = exp.artifact_checksum(artifact)
    artifact["reproducibility_checksum"] = checksum
    assert exp.artifact_checksum(artifact) == checksum


# REQ-REPORT-6575-ATOMIC: malformed local input fails closed without exceptions.
def test_json_reader_and_metadata_absence_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert exp.load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert exp.load_json(bad) == {}
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda _hf_id: None)
    rows = exp.build_fresh_metadata_rows()
    assert len(rows) == 3
    assert all(row["passed"] is False for row in rows)


# REQ-REPORT-6575-REDUCER: changing an emitted attack or protected row changes readiness.
def test_recompute_rejects_attack_or_protected_failure() -> None:
    artifact = _assemble()
    attacks = deepcopy(artifact["attack_rows"])
    attacks[0]["passed"] = False
    scores = exp.recompute_scores(
        context_rows=artifact["v570_ineligible_context_rows"],
        metadata_rows=artifact["model_revision_and_hash_receipts"],
        negative_rows=artifact["metadata_negative_fixture_rows"],
        runtime_receipts=artifact["process_and_gpu_receipts"],
        unload_rows=artifact["unload_and_recovery_rows"],
        method_rows=artifact["joint_sufficiency_method_replay_rows"],
        evidence_links=artifact["evidence_link_rows"],
        attack_rows=attacks,
        preconditions=artifact["preconditions_checked"],
        protected=_protected(unchanged=False),
        structural_verification=artifact["live_structural_verification"],
        duration_s=artifact["duration_s"],
    )
    assert scores["v571_flagship_evidence_ready_score"] == 0.0
    assert scores["joint_sufficiency_method_replay_ready_score"] == 1.0


# REQ-REPORT-6575-METADATA: non-content-addressed files still receive full hashes.
def test_file_hash_and_trusted_hash_fallback(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    plain = tmp_path / "plain.gguf"
    plain.write_bytes(b"GGUF fixture")

    assert exp.sha256_file(missing) == "missing"
    assert exp._trusted_hash_for_blob(plain) == exp.sha256_file(plain)  # noqa: SLF001


# REQ-REPORT-6575-METADATA: the fresh negative selector emits only preregistered rows.
def test_fresh_negative_selector_executes_required_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [
        {"unit_id": fixture_id, "passed": True, "observed_admitted": False}
        for fixture_id in (*exp.REQUIRED_NEGATIVE_FIXTURES, "extra_fixture")
    ]
    monkeypatch.setattr(exp, "build_negative_fixture_rows", lambda: source)

    rows = exp.build_fresh_negative_rows()

    assert [row["unit_id"] for row in rows] == list(exp.REQUIRED_NEGATIVE_FIXTURES)
    assert all(row["fresh"] and row["passed"] for row in rows)


# REQ-REPORT-6575-RECEIPTS / REQ-REPORT-6575-RESIDENCY:
# raw runtime recomputation calls every shipped identity/process/GPU/unload check.
def test_runtime_receipt_recomputation_requires_family_and_repeated_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _runtime_receipts()[0]
    receipt["process"]["selected_gpu"] = 0
    monkeypatch.setattr(exp.runtime, "identity_checks", lambda *_args: {"identity": True})
    monkeypatch.setattr(exp.runtime, "process_checks", lambda *_args: {"process": True})
    monkeypatch.setattr(exp.runtime, "gpu_checks", lambda *_args, **_kwargs: {"gpu": True})
    monkeypatch.setattr(exp.runtime, "unload_checks", lambda *_args: {"unload": True})

    assert exp.runtime_receipt_passes(receipt, _metadata_rows()) is True
    assert exp.runtime_receipt_passes({"repository_id": "substitute"}, _metadata_rows()) is False
    receipt["gpu_samples"] = [{"stage": "during"}]
    assert exp.runtime_receipt_passes(receipt, _metadata_rows()) is False


# REQ-REPORT-6575-RUNTIME: raw lifecycle containers become three hashed family receipts.
def test_runtime_receipt_builder_binds_family_and_unload_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = []
    unloads = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        processes.append(
            {
                "repository_id": hf_id,
                "raw_output_sha256": f"sha256:{str(index + 1) * 64}",
            }
        )
        unloads.append({"repository_id": hf_id, "recovery_complete": True})
    stage_rows = [
        {"repository_id": hf_id, "stage": "identity", "passed": True}
        for hf_id in exp.MANDATED_HF_IDS
    ]
    family_rows = [
        {"repository_id": hf_id, "family_admitted_score": 1.0} for hf_id in exp.MANDATED_HF_IDS
    ]
    monkeypatch.setattr(
        exp.runtime,
        "build_per_unit_rows",
        lambda *_args: (stage_rows, family_rows),
    )
    monkeypatch.setattr(exp.runtime, "unload_checks", lambda _row: {"recovered": True})

    receipts, fresh_unloads = exp.build_runtime_receipts(_metadata_rows(), processes, [], unloads)

    assert [row["repository_id"] for row in receipts] == list(exp.MANDATED_HF_IDS)
    assert all(row["passed"] and row["process"]["output_reused"] is False for row in receipts)
    assert all(row["passed"] for row in fresh_unloads)


# REQ-REPORT-6575-ATOMIC: a precondition stop with no runtime family is blocked.
def test_no_runtime_rows_with_failed_preconditions_is_blocked() -> None:
    artifact = _assemble(runtime_receipts=[], preconditions=_preconditions(ready=False))

    assert artifact["status"] == "blocked_v571_flagship_qualification"
    assert artifact["verdict_class"] == "blocked"


# REQ-REPORT-6575-ATOMIC: validator diagnostics cover malformed top-level structures.
def test_validator_covers_schema_policy_and_reducer_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp, "runtime_receipt_passes", lambda _row, _metadata: True)
    artifact = _assemble()
    broken = deepcopy(artifact)
    broken.pop("status")
    broken["inference_substrate"] = "wrong"
    broken["verifier_is_oracle"] = False
    broken["verdict_class"] = "positive"
    broken["model_specs"] = []
    broken["evidence_link_rows"][0]["passed"] = False
    broken["field_provenance"] = {}
    broken["model_revision_and_hash_receipts"][0]["fresh"] = False
    broken["v571_flagship_evidence_ready_score"] = 0.0
    broken["joint_sufficiency_method_replay_ready_score"] = 0.0
    broken["aggregate_row_recomputation"]["checks"] = {}

    errors = exp.validate_artifact(broken)

    assert "missing_required_fields" in errors
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_mismatch" in errors
    assert "verdict_class_outside_closed_set" in errors
    assert "positive_verdict_forbidden" in errors
    assert "model_order_or_family_mismatch" in errors
    assert "evidence_link_field_coverage_mismatch" in errors
    assert "field_provenance_incomplete" in errors
    assert "metadata_receipt_recomputation_failed" in errors
    assert "aggregate_check_recomputation_mismatch" in errors

    score_mismatch = deepcopy(artifact)
    score_mismatch["v571_flagship_evidence_ready_score"] = 0.0
    score_mismatch["joint_sufficiency_method_replay_ready_score"] = 0.0
    score_errors = exp.validate_artifact(score_mismatch)
    assert "flagship_ready_score_mismatch" in score_errors
    assert "method_ready_score_mismatch" in score_errors

    not_lists = deepcopy(artifact)
    not_lists["attack_rows"] = {}
    assert "row_container_not_list" in exp.validate_artifact(not_lists)


# REQ-REPORT-6575-ATOMIC: the normal CLI branch reports both exact gate fields.
def test_main_run_branch_prints_terminal_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _root, _date: {
            "status": "complete_v571_flagship_qualification_ready",
            "v571_flagship_evidence_ready_score": 1.0,
            "joint_sufficiency_method_replay_ready_score": 1.0,
        },
    )

    assert exp.main(["--date", "20260824"]) == 0
    output = capsys.readouterr().out
    assert "v571_flagship_evidence_ready_score" in output
    assert "joint_sufficiency_method_replay_ready_score" in output
