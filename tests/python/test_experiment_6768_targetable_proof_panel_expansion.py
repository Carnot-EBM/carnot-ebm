"""Focused tests for the answer-blind exact-invalid proof panel.

Spec refs: REQ-VERIFY-6768, SCENARIO-VERIFY-6768-*, REQ-REPORT-6768,
and SCENARIO-REPORT-6768-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from carnot import experiment_6768_targetable_proof_panel_expansion as exp


def _frozen_inputs() -> tuple[dict, dict, dict]:
    """Load the three read-only inputs used by the focused fixture tests."""

    return tuple(  # type: ignore[return-value]
        exp.load_json_object(exp.REPO_ROOT / path)
        for path in (
            exp.UPSTREAM_REPLAY_PATH,
            exp.UPSTREAM_PROPOSAL_PATH,
            exp.UPSTREAM_STREAM_PATH,
        )
    )


def _sources() -> list[dict]:
    """Project the real frozen source rows through the answer-blind boundary."""

    return exp.project_source_candidates(*_frozen_inputs())


def _evaluated_rows() -> list[dict]:
    """Build every deterministic mutation without running artifact publication."""

    return [
        exp.evaluate_mutation(source, mutation)
        for source in _sources()
        for mutation in exp.build_operator_mutations(source)
    ]


def test_req_verify_6768_preconditions_freeze_the_21_source_rows() -> None:
    """SCENARIO-VERIFY-6768-BLOCKED checks each frozen source invariant."""

    replay, proposal, stream = _frozen_inputs()
    ready = exp.evaluate_preconditions(replay, proposal, stream)

    assert ready["all_passed"] is True
    assert exp.first_failed_check(ready)["check"] == "all_preconditions"
    assert [row["check"] for row in ready["checks"]] == [
        "exp6755_transport_reparse_ready",
        "exp6755_row_count",
        "exp6755_unique_row_ids",
        "exp6755_raw_and_normalized_hashes",
        "exp6755_targetable_source_rows",
        "exp6745_artifact_hash_lineage",
        "exp6745_source_row_lineage",
        "exp6744_cnf_lineage",
    ]

    cases = [
        (
            "exp6755_transport_reparse_ready",
            lambda r, _p, _s: r.__setitem__("transport_reparse_ready", False),
        ),
        ("exp6755_row_count", lambda r, _p, _s: r["rows"].pop()),
        (
            "exp6755_unique_row_ids",
            lambda r, _p, _s: r["rows"][1].__setitem__("row_id", r["rows"][0]["row_id"]),
        ),
        (
            "exp6755_raw_and_normalized_hashes",
            lambda r, _p, _s: r["rows"][0].__setitem__("normalized_output_sha256", "bad"),
        ),
        (
            "exp6755_targetable_source_rows",
            lambda r, _p, _s: next(
                row
                for row in r["rows"]
                if row["grammar_failures"]["environment_grammar_targetable"]
            )["grammar_failures"].__setitem__("environment_grammar_targetable", False),
        ),
        (
            "exp6745_artifact_hash_lineage",
            lambda r, _p, _s: r.__setitem__("source_artifact_sha256", "bad"),
        ),
        (
            "exp6745_source_row_lineage",
            lambda r, _p, _s: r["rows"][0].__setitem__("source_artifact_row_sha256", "bad"),
        ),
        (
            "exp6744_cnf_lineage",
            lambda r, _p, _s: r["rows"][0]["source_row"].__setitem__(
                "exact_stream_row_sha256", "bad"
            ),
        ),
    ]
    for expected, mutation in cases:
        changed = deepcopy((replay, proposal, stream))
        mutation(*changed)
        failed = exp.first_failed_check(exp.evaluate_preconditions(*changed))
        assert failed["check"] == expected
        assert failed["passed"] is False


def test_scenario_verify_6768_leakage_projection_ignores_forbidden_authority() -> None:
    """SCENARIO-VERIFY-6768-LEAKAGE makes forbidden authority inert."""

    replay, proposal, stream = _frozen_inputs()
    baseline = exp.project_source_candidates(replay, proposal, stream)
    changed = deepcopy((replay, proposal, stream))
    for row in changed[0]["rows"]:
        row["exact_outcome"] = "changed"
        row["exact_valid"] = "changed"
        row["solver_trace"] = "changed"
        row["ground_truth_certificate"] = "changed"
        row["answer_label"] = "changed"
    for row in changed[1]["rows"]:
        row["diagnosis"] = "changed"
        row["answer_label"] = "changed"
    for row in changed[2]["rows"]:
        row["label"] = "changed"
        row["solver_trace"] = "changed"

    assert exp.project_source_candidates(*changed) == baseline
    assert len(baseline) == exp.EXPECTED_TARGETABLE_SOURCE_COUNT == 21
    assert all(set(source).isdisjoint(exp.FORBIDDEN_FEATURE_NAMES) for source in baseline)
    assert exp.FUTURE_OR_ANSWER_FEATURES_READ == []


def test_scenario_verify_6768_mutation_operators_are_local_and_exclusive() -> None:
    """SCENARIO-VERIFY-6768-MUTATION preregisters six one-region operators."""

    source = _sources()[0]
    mutations = exp.build_operator_mutations(source)

    assert [mutation["error_class"] for mutation in mutations] == list(exp.ERROR_CLASSES)
    assert len({mutation["after_certificate"] for mutation in mutations}) == len(exp.ERROR_CLASSES)
    for mutation in mutations:
        assert mutation["mutation_operator"].startswith("answer_blind_")
        assert mutation["before_certificate"] != mutation["after_certificate"]
        assert mutation["target_region"]["attributable"] is True
        assert mutation["target_region"]["smallest_responsible_region"] is True
        parsed = exp.parse_counterfactual_certificate(mutation["after_certificate"])
        detected = exp.detect_error_classes(source["cnf"], parsed)
        assert parsed["parser_status"] == "parseable"
        assert detected == [mutation["error_class"]]


@pytest.mark.parametrize(
    "text",
    ["", "ABSTAIN", "MAYBE x1=0", "SAT", "SAT y1=0", "UNSAT", "UNSAT x1"],
)
def test_req_verify_6768_structural_parser_rejects_non_certificate_text(text: str) -> None:
    """REQ-VERIFY-6768 keeps structural parseability strict and explicit."""

    receipt = exp.parse_counterfactual_certificate(text)

    assert receipt["parser_status"] == "malformed"
    assert receipt["parse_failure"]


def test_req_verify_6768_rejects_multi_class_or_unattributable_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6768 rejects each mutation shape outside the panel contract."""

    source = _sources()[0]
    mutation = exp.build_operator_mutations(source)[0]
    assert exp.detect_error_classes(source["cnf"], {"parser_status": "malformed"}) == []
    duplicate_core = exp.parse_counterfactual_certificate("UNSAT c1,c1")
    assert exp.detect_error_classes(source["cnf"], duplicate_core) == ["duplicate_evidence"]

    malformed = deepcopy(mutation)
    malformed["after_certificate"] = "not a certificate"
    with pytest.raises(ValueError, match="unparsable_mutation"):
        exp.evaluate_mutation(source, malformed)

    wrong_class = deepcopy(mutation)
    wrong_class["error_class"] = "missing_evidence"
    with pytest.raises(ValueError, match="mutation_error_classes"):
        exp.evaluate_mutation(source, wrong_class)

    bad_region = deepcopy(mutation)
    bad_region["target_region"] = {}
    with pytest.raises(ValueError, match="unattributable_target_region"):
        exp.evaluate_mutation(source, bad_region)

    monkeypatch.setattr(
        exp.frozen,
        "exact_check_constraints",
        lambda _cnf, _constraints: {
            "attempted": True,
            "authority_available": True,
            "valid": True,
            "reason": "forced_fixture",
            "checked_assignment_count": 1,
        },
    )
    with pytest.raises(ValueError, match="exact_valid_mutation"):
        exp.evaluate_mutation(source, mutation)


def test_scenario_verify_6768_exact_receipts_cover_all_error_classes() -> None:
    """SCENARIO-VERIFY-6768-MUTATION keeps every mutation exact-invalid."""

    source = _sources()[0]
    rows = [
        exp.evaluate_mutation(source, mutation) for mutation in exp.build_operator_mutations(source)
    ]

    assert [row["error_class"] for row in rows] == list(exp.ERROR_CLASSES)
    assert all(row["parser_receipt"]["parser_status"] == "parseable" for row in rows)
    assert all(row["detected_error_classes"] == [row["error_class"]] for row in rows)
    assert all(row["exact_failure_receipt"]["attempted"] is True for row in rows)
    assert all(row["exact_failure_receipt"]["valid"] is False for row in rows)
    assert all(row["exact_valid"] is False for row in rows)
    assert all(row["source_problem_unchanged"] is True for row in rows)
    assert all(row["encoder_a_receipt"]["attempted"] is True for row in rows)
    assert all(row["encoder_b_receipt"]["attempted"] is True for row in rows)
    non_binary = next(row for row in rows if row["error_class"] == "non_binary_value")
    assert non_binary["encoder_a_receipt"]["accepted"] is False
    assert non_binary["encoder_b_receipt"]["accepted"] is False
    assert non_binary["exact_failure_receipt"]["reason"] == "assignment_value_invalid"
    assert all(
        row["encoder_a_receipt"]["accepted"] is True
        and row["encoder_b_receipt"]["accepted"] is True
        for row in rows
        if row["error_class"] != "non_binary_value"
    )


def test_req_report_6768_rows_derive_coverage_and_relabel_receipts() -> None:
    """SCENARIO-REPORT-6768-READY derives panel evidence only from rows."""

    rows = _evaluated_rows()
    reduction = exp.recompute_row_aggregates(rows)
    pair_receipts = exp.build_relabel_receipts(rows)

    assert len(rows) == 21 * 6 == 126
    assert reduction["targetable_row_count"] == 126
    assert reduction["counts_by_family"] == {
        "expander_tseitin": 24,
        "ladder_tseitin": 72,
        "pigeonhole_anchor": 30,
    }
    assert reduction["counts_by_error_class"] == {
        error_class: 21 for error_class in exp.ERROR_CLASSES
    }
    assert reduction["exact_valid_mutations"] == 0
    assert reduction["duplicate_rows"] == 0
    assert len(pair_receipts) == 4 * 6
    assert all(receipt["same_split"] is True for receipt in pair_receipts)
    assert all(receipt["pair_invariance_passed"] is True for receipt in pair_receipts)
    assert all(row["row_sha256"] == exp.row_checksum(row) for row in rows)


def test_req_report_6768_cold_replay_detects_receipt_tampering() -> None:
    """REQ-REPORT-6768 cold-replays parser, encoders, and exact checks."""

    rows = _evaluated_rows()[:6]
    receipt = exp.cold_replay_rows(rows, producer_pid=os.getpid() + 1)

    assert receipt["fresh_process"] is True
    assert receipt["all_passed"] is True
    assert receipt["replayed_row_count"] == 6
    assert receipt["mismatches"] == []

    changed = deepcopy(rows)
    changed[0]["exact_failure_receipt"]["reason"] = "tampered"
    failed = exp.cold_replay_rows(changed, producer_pid=os.getpid() + 1)
    assert failed["all_passed"] is False
    assert failed["mismatches"][0]["row_id"] == changed[0]["row_id"]

    invalid = deepcopy(rows)
    invalid[0].pop("source_candidate")
    rejected = exp.cold_replay_rows(invalid, producer_pid=os.getpid() + 1)
    assert rejected["mismatches"][0]["reason"].startswith("KeyError:")


def test_req_report_6768_cold_worker_failure_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6768 fails closed when the fresh worker cannot publish."""

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7, stderr="worker failed"),
    )

    receipt = exp.run_cold_replay([])

    assert receipt["all_passed"] is False
    assert receipt["mismatches"] == [{"row_id": None, "reason": "cold_worker_exit_7"}]
    assert receipt["stderr"] == "worker failed"


def test_req_report_6768_build_validate_and_reject_drift() -> None:
    """SCENARIO-REPORT-6768-READY validates every row-derived fixture gate."""

    replay, proposal, stream = _frozen_inputs()
    rows = _evaluated_rows()
    preconditions = exp.evaluate_preconditions(replay, proposal, stream)
    cold = exp.cold_replay_rows(rows, producer_pid=os.getpid() + 1)
    artifact = exp.build_artifact(
        date="20260830",
        duration_s=0.25,
        rows=rows,
        source_artifact_sha256=exp.sha256_file(exp.REPO_ROOT / exp.UPSTREAM_REPLAY_PATH),
        source_proposal_artifact_sha256=exp.sha256_file(exp.REPO_ROOT / exp.UPSTREAM_PROPOSAL_PATH),
        source_stream_artifact_sha256=exp.sha256_file(exp.REPO_ROOT / exp.UPSTREAM_STREAM_PATH),
        preconditions=preconditions,
        cold_replay_receipt=cold,
    )

    assert artifact["targetable_panel_ready"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["future_or_answer_features_read"] == []
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []

    cases = [
        ("aggregate_recomputation_mismatch", "targetable_row_count", 1),
        ("readiness_gate_mismatch", "targetable_panel_ready", False),
        ("verifier_is_oracle_mismatch", "verifier_is_oracle", True),
        ("verdict_class_invalid", "verdict_class", "invalid"),
    ]
    for expected, field, value in cases:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    assert exp.validate_artifact({}) == [
        "missing_required_fields:" + ",".join(sorted(exp.ARTIFACT_FIELDS))
    ]
    changed = deepcopy(artifact)
    changed["field_principles"].pop("title")
    changed["inference_substrate"] = "wrong"
    changed["future_or_answer_features_read"] = ["answer_label"]
    changed["reproducibility_receipt"] = []
    assert set(exp.validate_artifact(changed)) >= {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "answer_feature_leakage",
        "reproducibility_checksum_mismatch",
    }

    drift_cases = [
        ("source_row_id_recomputation_mismatch", "source_targetable_row_ids", []),
        ("operator_recomputation_mismatch", "mutation_operators", {}),
        ("relabel_recomputation_mismatch", "proof_preserving_relabel_receipts", []),
    ]
    for expected, field, value in drift_cases:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["rows"][0]["after_certificate"] = "SAT x1=0"
    assert "row_checksum_mismatch" in exp.validate_artifact(changed)
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_scenario_report_6768_blocked_artifact_is_atomic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6768-BLOCKED publishes the first failed input check."""

    blocked = exp.build_blocked_artifact(
        date="20260830",
        duration_s=0.1,
        failed_check="fixture_check",
        expected=True,
        observed=False,
        source_artifact_sha256="sha256:source",
    )
    assert blocked["status"] == "complete_blocked_targetable_panel"
    assert blocked["honest_verdict"].startswith("complete_blocked_targetable_panel")
    assert blocked["gate_check_summary"]["failed_check"] == "fixture_check"
    assert blocked["rows"] == []
    assert blocked["targetable_panel_ready"] is False
    assert exp.validate_artifact(blocked) == []

    results = tmp_path / "results"
    results.mkdir()
    (results / exp.UPSTREAM_REPLAY_PATH.name).write_text("not json", encoding="utf-8")
    artifact = exp.run("20260830", tmp_path)
    assert artifact["gate_check_summary"]["failed_check"] == "exp6755_json_object"
    assert json.loads((results / exp.RESULT_PATH.name).read_text()) == artifact

    bad = deepcopy(blocked)
    bad["rows"] = [{}]
    bad["verdict_class"] = "partial"
    bad["honest_verdict"] = "wrong"
    bad["targetable_panel_ready"] = True
    assert set(exp.validate_artifact(bad)) >= {
        "blocked_rows_invalid",
        "blocked_verdict_class_mismatch",
        "blocked_verdict_prefix_mismatch",
        "blocked_readiness_mismatch",
        "reproducibility_checksum_mismatch",
    }
    with pytest.raises(ValueError):
        exp.write_json_atomic(tmp_path / "bad.json", bad)


def test_scenario_verify_6768_run_blocks_each_late_input_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6768-BLOCKED stops at lineage or mutation failure."""

    replay, proposal, stream = _frozen_inputs()

    proposal_root = tmp_path / "proposal"
    proposal_results = proposal_root / "results"
    proposal_results.mkdir(parents=True)
    (proposal_results / exp.UPSTREAM_REPLAY_PATH.name).write_text(json.dumps(replay))
    (proposal_results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text("[]")
    proposal_block = exp.run("20260830", proposal_root)
    assert proposal_block["gate_check_summary"]["failed_check"] == "exp6745_json_object"

    stream_root = tmp_path / "stream"
    stream_results = stream_root / "results"
    stream_results.mkdir(parents=True)
    (stream_results / exp.UPSTREAM_REPLAY_PATH.name).write_text(json.dumps(replay))
    (stream_results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text(json.dumps(proposal))
    (stream_results / exp.UPSTREAM_STREAM_PATH.name).write_text("[]")
    stream_block = exp.run("20260830", stream_root)
    assert stream_block["gate_check_summary"]["failed_check"] == "exp6744_json_object"

    gate_root = tmp_path / "gate"
    gate_results = gate_root / "results"
    gate_results.mkdir(parents=True)
    changed_replay = deepcopy(replay)
    changed_replay["transport_reparse_ready"] = False
    (gate_results / exp.UPSTREAM_REPLAY_PATH.name).write_text(json.dumps(changed_replay))
    (gate_results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text(json.dumps(proposal))
    (gate_results / exp.UPSTREAM_STREAM_PATH.name).write_text(json.dumps(stream))
    gate_block = exp.run("20260830", gate_root)
    assert gate_block["gate_check_summary"]["failed_check"] == ("exp6755_transport_reparse_ready")

    mutation_root = tmp_path / "mutation"
    mutation_results = mutation_root / "results"
    mutation_results.mkdir(parents=True)
    for source in (
        exp.UPSTREAM_REPLAY_PATH,
        exp.UPSTREAM_PROPOSAL_PATH,
        exp.UPSTREAM_STREAM_PATH,
    ):
        shutil.copy2(exp.REPO_ROOT / source, mutation_results / source.name)
    monkeypatch.setattr(
        exp,
        "evaluate_mutation",
        lambda _source, _mutation: (_ for _ in ()).throw(ValueError("fixture")),
    )
    mutation_block = exp.run("20260830", mutation_root)
    assert mutation_block["gate_check_summary"]["failed_check"] == "mutation_expansion"


def test_req_report_6768_actual_run_writes_fresh_process_panel(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6768-READY publishes the full cold-replayed fixture."""

    results = tmp_path / "results"
    results.mkdir()
    for source in (
        exp.UPSTREAM_REPLAY_PATH,
        exp.UPSTREAM_PROPOSAL_PATH,
        exp.UPSTREAM_STREAM_PATH,
    ):
        shutil.copy2(exp.REPO_ROOT / source, results / source.name)

    artifact = exp.run("20260830", tmp_path)
    written = exp.load_json_object(results / exp.RESULT_PATH.name)

    assert written == artifact
    assert artifact["targetable_panel_ready"] is True
    assert artifact["targetable_row_count"] == 126
    assert artifact["cold_replay_receipt"]["fresh_process"] is True
    assert (
        artifact["cold_replay_receipt"]["cold_pid"]
        != artifact["cold_replay_receipt"]["producer_pid"]
    )
    assert artifact["cold_replay_receipt"]["all_passed"] is True
    assert exp.validate_artifact(artifact) == []


def test_req_report_6768_spec_and_cli_contract(tmp_path: Path) -> None:
    """REQ-REPORT-6768 keeps both anchors and the cold worker CLI explicit."""

    verify_spec = (exp.REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md").read_text()
    report_spec = (exp.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text()
    payload = tmp_path / "payload.json"
    output = tmp_path / "cold.json"
    payload.write_text(json.dumps({"producer_pid": os.getpid() + 1, "rows": []}))

    assert "REQ-VERIFY-6768" in verify_spec
    assert "SCENARIO-VERIFY-6768-MUTATION" in verify_spec
    assert "REQ-REPORT-6768" in report_spec
    assert "SCENARIO-REPORT-6768-READY" in report_spec
    assert exp.parse_args([]).date == "20260830"
    assert exp.parse_args(["--date", "20260901"]).date == "20260901"
    assert (
        exp.parse_args(
            ["--cold-replay-input", str(payload), "--cold-replay-output", str(output)]
        ).cold_replay_input
        == payload
    )
    assert exp.cold_replay_main(payload, output) == 0
    assert exp.load_json_object(output)["fresh_process"] is True
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    array = tmp_path / "array.json"
    array.write_text("[]")
    with pytest.raises(TypeError, match="JSON object required"):
        exp.load_json_object(array)
