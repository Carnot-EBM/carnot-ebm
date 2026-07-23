"""Tests for Exp5826 out-of-template chronological constraint stream.

Spec refs: REQ-LEARN-5826, SCENARIO-LEARN-5826-STREAM,
SCENARIO-LEARN-5826-OUT-OF-TEMPLATE, SCENARIO-LEARN-5826-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5825_certified_adaptive_memory_contract as contract
from carnot import experiment_5826_out_of_template_constraint_stream as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5826_out_of_template_constraint_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5826_out_of_template_constraint_stream.py "
    "-m pytest tests/python/test_experiment_5826_out_of_template_constraint_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5826_out_of_template_constraint_stream.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5826_out_of_template_constraint_stream.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )


def _run_stream(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    rows = mod.read_row_file(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)
    return artifact, rows


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.keys()) + [item for sub in value.values() for item in _walk(sub)]
    if isinstance(value, list):
        return [item for sub in value for item in _walk(sub)]
    return [value]


def test_req_learn_5826_spec_declares_stream_contract_fields_and_principles() -> None:
    """REQ-LEARN-5826: OpenSpec anchors the stream contract and required fields."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5826") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5826",
        "SCENARIO-LEARN-5826-STREAM",
        "SCENARIO-LEARN-5826-OUT-OF-TEMPLATE",
        "SCENARIO-LEARN-5826-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`constraint_event_stream_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5826_stream_rows_are_balanced_chronological_and_replayable(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5826-STREAM: ready rows meet planned cells and replay hashes."""

    artifact, rows = _run_stream(tmp_path)
    rerun = mod.run(
        result_path=tmp_path / "rerun.json",
        row_file_path=tmp_path / "rerun.rows.jsonl",
        preconditions_checked=_preconditions(tmp_path / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["constraint_event_stream_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["constraint_event_stream_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["duration_s"] >= 0.0
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["row_file_and_sha256"]["sha256"] == mod.sha256_file(
        tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    )
    assert artifact["row_file_and_sha256"]["sha256"] == rerun["row_file_and_sha256"]["sha256"]
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]

    expected_rows = len(mod.PRIMARY_FAMILIES) * len(mod.CHANGE_ORDER) * mod.MIN_UNITS_PER_CELL
    manifest = artifact["stream_manifest"]
    assert len(rows) == expected_rows == manifest["row_count"]
    assert manifest["family_count"] == 4
    assert set(manifest["families"]) == set(mod.PRIMARY_FAMILIES)
    assert manifest["changes"] == list(mod.CHANGE_ORDER)
    assert all(count == mod.MIN_UNITS_PER_CELL for count in manifest["cell_counts"].values())
    assert manifest["minimum_science_units_per_primary_cell"] == mod.MIN_UNITS_PER_CELL
    assert manifest["hardness_surface_crossing"]["all_cells_have_all_pairs"] is True
    assert set(manifest["proof_preserving_surfaces"]) == set(mod.PROOF_PRESERVING_SURFACES)
    assert manifest["audit_summary"]["sat_unsat_mix_ok"] is True
    assert manifest["audit_summary"]["causal_pair_availability_ok"] is True
    assert manifest["audit_summary"]["checkpoint_atomicity_ok"] is True

    receipts = artifact["chronology_and_change_receipts"]
    assert receipts["all_passed"] is True
    for family in mod.PRIMARY_FAMILIES:
        assert receipts["family_change_order"][family] == list(mod.CHANGE_ORDER)
        assert receipts["addition_count_by_family"][family] == mod.MIN_UNITS_PER_CELL
        assert receipts["supersession_count_by_family"][family] == mod.MIN_UNITS_PER_CELL
        assert receipts["recurrence_count_by_family"][family] == mod.MIN_UNITS_PER_CELL

    assert [row["chronology_index"] for row in rows] == list(range(len(rows)))
    assert all(row["row_hash"] == artifact["row_file_and_sha256"]["row_hashes"][row["row_id"]] for row in rows)
    assert all(row["checkpoint_receipt"]["atomic_commit"] is True for row in rows)
    assert all(row["parent_state_hash"].startswith("sha256:") for row in rows)


def test_scenario_learn_5826_targets_are_machine_checked_out_of_template(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5826-OUT-OF-TEMPLATE: signatures are absent from Exp5762."""

    artifact, rows = _run_stream(tmp_path)
    library = mod.frozen_template_signature_receipt()

    assert library["signature_count"] > 0
    assert artifact["out_of_template_witnesses"]["all_targets_out_of_template"] is True
    assert artifact["out_of_template_witnesses"]["machine_checked"] is True
    assert artifact["out_of_template_witnesses"]["target_count"] == len(rows)
    assert artifact["out_of_template_witnesses"]["expressible_target_count"] == 0
    assert artifact["out_of_template_witnesses"]["library_signature_hash"] == library[
        "signature_root_hash"
    ]
    assert all(row["out_of_template_witness"]["absent_from_frozen_library"] is True for row in rows)
    assert all(row["out_of_template_witness"]["machine_checked"] is True for row in rows)

    for family in mod.PRIMARY_FAMILIES:
        signature = mod.target_signature_for_family(family)
        assert mod.signature_in_frozen_library(signature, library["signatures"]) is False

    generic_signature = {"relation": "equals", "arity": 1, "composition": "atomic"}
    assert mod.signature_in_frozen_library(generic_signature, library["signatures"]) is True


def test_req_learn_5826_exact_receipts_events_and_leakage_boundaries(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5826: exact evidence is emitted without hidden-label leakage."""

    artifact, rows = _run_stream(tmp_path)
    event_types = {event["event_type"] for row in rows for event in row["canonical_events"]}
    states = [state for row in rows for state in row["canonical_states"]]
    events = [event for row in rows for event in row["canonical_events"]]

    assert contract.validate_event_stream(events, states) == []
    assert event_types >= {
        "observation",
        "exact_membership_outcome",
        "minimal_core_evidence",
        "protected_prefix_replay",
        "sealed_future_evaluation",
        "constraint_birth",
        "supersession",
        "recurrence",
    }
    assert artifact["exact_query_and_core_receipts"]["all_exact_validators_agree"] is True
    assert artifact["exact_query_and_core_receipts"]["sat_count"] > 0
    assert artifact["exact_query_and_core_receipts"]["unsat_count"] > 0
    assert artifact["exact_query_and_core_receipts"]["minimal_evidence_count"] == len(rows)
    assert artifact["protected_prefix_receipts"]["all_passed"] is True
    assert artifact["sealed_future_batch_receipts"]["all_future_suffixes_sealed"] is True
    assert artifact["sealed_future_batch_receipts"]["future_label_leakage_count"] == 0
    assert artifact["leakage_audit"]["leakage_count"] == 0
    assert artifact["leakage_audit"]["ground_truth_structure_sealed"] is True
    assert artifact["leakage_audit"]["future_labels_sealed"] is True
    assert artifact["leakage_audit"]["llm_generated_text_count"] == 0
    assert artifact["sample_size_and_justification"]["minimum_units_per_primary_cell"] == 30

    leak_tokens = {
        "target_constraint",
        "ground_truth_structure",
        "exact_label",
        "future_label",
        "sealed_ground_truth",
    }
    learner_values = [str(item) for row in rows for item in _walk(row["learner_view"])]
    assert not leak_tokens.intersection(learner_values)
    assert all(row["ground_truth_structure_seal"].startswith("sha256:") for row in rows)
    assert all("plaintext" not in row["ground_truth_structure_boundary"] for row in rows)
    assert all(row["sealed_future_suffix"]["future_labels_visible_to_learner"] is False for row in rows)
    assert all(row["exact_receipt"]["primary"]["validators_agree"] is True for row in rows)
    assert all(row["exact_receipt"]["independent"]["validators_agree"] is True for row in rows)


def test_scenario_learn_5826_fail_closed_for_preconditions_exits_and_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5826-FAIL-CLOSED: blocked states cannot report readiness."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["constraint_event_stream_ready_score"] == 0.0
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    artifact, rows = _run_stream(tmp_path / "ready")
    failed_exits = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "failed-exits"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    assert failed_exits["status"] == "blocked"
    assert failed_exits["constraint_event_stream_ready_score"] == 0.0
    assert failed_exits["honest_verdict"] == "blocked: failed_test_exit_codes"

    bad_rows = deepcopy(rows)
    bad_rows[0]["row_hash"] = mod.sha256_text("tamper")
    with pytest.raises(mod.StreamReplayError, match="row_hash mismatch"):
        mod.verify_row_file(bad_rows, artifact)

    leaky = deepcopy(rows)
    leaky[0]["learner_view"]["ground_truth_structure"] = {"relation": "leak"}
    audit = mod.leakage_audit_for_rows(leaky)
    assert audit["leakage_count"] > 0
    assert audit["ground_truth_structure_sealed"] is False

    monkeypatch.setattr(
        mod,
        "_load_upstreams",
        lambda root: (_ for _ in ()).throw(ValueError("corrupt")),
    )
    corrupt = mod.collect_preconditions(
        result_path=tmp_path / "corrupt.json",
        row_file_path=tmp_path / "corrupt.rows.jsonl",
        memory_probe=lambda: {"available_mb": 0, "required_mb": 512, "ok": False},
        disk_probe=lambda root: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    assert set(corrupt["blocked_reasons"]) >= {
        "corrupt_upstream_artifact",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }


def test_req_learn_5826_validation_rejects_artifact_drift_and_wrapper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5826: validators reject stale score, checksum, and wrapper drift."""

    artifact, _rows = _run_stream(tmp_path)

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required artifact fields"),
        (lambda item: item.update({"inference_substrate": "live_llm_inference"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance"),
        (lambda item: item["test_exit_codes"].update({TEST_COMMAND: 1}), "constraint_event_stream_ready_score"),
        (lambda item: item["leakage_audit"].update({"leakage_count": 1}), "constraint_event_stream_ready_score"),
        (lambda item: item["row_file_and_sha256"].update({"sha256": mod.sha256_text("wrong")}), "row_file_and_sha256"),
        (lambda item: item.update({"honest_verdict": "ready"}), "honest_verdict"),
        (lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}), "reproducibility_checksum"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" in bad and match != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    complete_with_blocked_verdict = deepcopy(artifact)
    complete_with_blocked_verdict["honest_verdict"] = "blocked: forced"
    complete_with_blocked_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        complete_with_blocked_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(complete_with_blocked_verdict)

    monkeypatch.setattr(
        mod,
        "build_and_write_artifacts",
        lambda **kwargs: {"test_commands": kwargs["test_commands"]},
    )
    assert mod.run(write=True)["test_commands"] == list(mod.DEFAULT_TEST_COMMANDS)
    called: list[bool] = []
    monkeypatch.setattr(mod, "run", lambda: called.append(True))
    assert mod.main() == 0
    assert called == [True]


def test_req_learn_5826_low_level_edge_cases_are_exercised(tmp_path: Path) -> None:
    """REQ-LEARN-5826: replay helpers fail closed on malformed or stale evidence."""

    artifact, rows = _run_stream(tmp_path)

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("1\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{"ok": True}]
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(scalar_jsonl)
    assert mod.read_row_file(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions()["preconditions_ready"] is True

    assert mod._signature_for_constraint(
        {"type": "forbid_assignment", "assignment": {"A": 1, "B": 2}}
    ) == {"relation": "forbid_assignment", "arity": 2, "composition": "exact_assignment_tuple"}
    assert mod._signature_for_constraint(
        {"relation": "new_relation", "arity": 4, "composition": "nested"}
    ) == {"relation": "new_relation", "arity": 4, "composition": "nested"}
    assert mod._flatten_for_leakage(["alpha", {"beta": "gamma"}]) == [
        "alpha",
        "beta",
        "gamma",
    ]

    with pytest.raises(mod.StreamReplayError, match="duplicate row_id"):
        mod.verify_row_file([rows[0], rows[0]], artifact)

    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_and_sha256"]["row_hashes"][rows[0]["row_id"]] = mod.sha256_text(
        "wrong-row"
    )
    with pytest.raises(mod.StreamReplayError, match="artifact row_hash mismatch"):
        mod.verify_row_file(rows, bad_artifact)

    bad_count = deepcopy(artifact)
    bad_count["row_file_and_sha256"]["row_count"] = 0
    with pytest.raises(mod.StreamReplayError, match="row count mismatch"):
        mod.verify_row_file(rows, bad_count)

    bad_sha = deepcopy(artifact)
    bad_sha["row_file_and_sha256"]["sha256"] = mod.sha256_text("wrong-file")
    with pytest.raises(mod.StreamReplayError, match="row_file_sha256 mismatch"):
        mod.verify_row_file(rows, bad_sha)

    bad_event_rows = deepcopy(rows)
    event = bad_event_rows[0]["canonical_events"][1]
    event["oracle_provenance"]["hidden_label_access"] = True
    event["event_hash"] = contract.canonical_event_hash(event)
    event["event_id"] = contract.expected_event_id(event)
    bad_event_rows[0]["row_hash"] = mod._row_hash(bad_event_rows[0])
    bad_event_artifact = deepcopy(artifact)
    bad_event_artifact["row_file_and_sha256"] = mod._row_file_receipt(
        bad_event_rows,
        mod.rows_to_jsonl(bad_event_rows),
    )
    with pytest.raises(mod.StreamReplayError, match="hidden_science_label_access"):
        mod.verify_row_file(bad_event_rows, bad_event_artifact)

    for mutate, expected in (
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"row_file_and_sha256": {}}), "row_file_and_sha256"),
        (lambda item: item["leakage_audit"].update({"leakage_count": 2}), "leakage_audit"),
        (lambda item: item["stream_manifest"].update({"row_count": 0}), "constraint_event_stream_ready_score"),
    ):
        blocked_probe = deepcopy(artifact)
        mutate(blocked_probe)
        assert expected in mod.blocked_reasons(blocked_probe)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "ready"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_provenance = deepcopy(artifact)
    invalid_provenance["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(invalid_provenance)

    status_mismatch = deepcopy(artifact)
    status_mismatch["status"] = "blocked"
    status_mismatch["reproducibility_checksum"] = mod.reproducibility_checksum(status_mismatch)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status_mismatch)

    blocked_with_bad_verdict = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "bad-verdict"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 1},
    )
    blocked_with_bad_verdict["honest_verdict"] = "ready"
    blocked_with_bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_with_bad_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_with_bad_verdict)

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        row_file_path=tmp_path / "no-write.rows.jsonl",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "complete"
    assert not (tmp_path / "no-write.json").exists()
