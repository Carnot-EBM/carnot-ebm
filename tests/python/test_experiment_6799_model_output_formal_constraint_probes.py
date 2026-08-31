"""Tests for frozen authentic-output formal constraint probes.

Spec refs: REQ-VERIFY-6799 and SCENARIO-VERIFY-6799-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6799_model_output_formal_constraint_probes as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"


@pytest.fixture(scope="module")
def sources() -> tuple[dict, dict]:
    """Load the two frozen source artifacts once for read-only tests."""

    return exp.load_source_artifacts(repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def eligible_cases(sources: tuple[dict, dict]) -> list[dict]:
    """Select every authentic row that supports strict paired operations."""

    return exp.select_eligible_cases(*sources)


@pytest.fixture(scope="module")
def groups(eligible_cases: list[dict]) -> list[dict]:
    """Build all paired graph groups once because exact enumeration is deterministic."""

    return exp.build_probe_groups(eligible_cases)


@pytest.fixture(scope="module")
def rows(groups: list[dict]) -> list[dict]:
    """Build valid and hard-negative rows for each transformation graph."""

    return exp.build_rows(groups)


def test_req_verify_6799_spec_declares_the_real_output_fixture_contract() -> None:
    """REQ-VERIFY-6799 anchors the source, operation, split, and attack rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-6799") :]
    for marker in (
        "REQ-VERIFY-6799",
        "SCENARIO-VERIFY-6799-GENERATOR",
        "SCENARIO-VERIFY-6799-ENUMERATION",
        "SCENARIO-VERIFY-6799-SPLITS",
        "SCENARIO-VERIFY-6799-MUTATIONS",
        "SCENARIO-VERIFY-6799-REPLAY",
        "SCENARIO-VERIFY-6799-BLOCKED",
        "at least 96 eligible SAT source cases",
        "complete_blocked_model_output_probe_fixture",
        "model_output_constraint_probe_ready",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in exp.STANDARD_ARTIFACT_FIELDS


def test_scenario_verify_6799_preconditions_require_authentic_complete_sources(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6799-BLOCKED rejects source or mandated-model drift."""

    summary = exp.evaluate_preconditions(repo_root=REPO_ROOT)
    assert summary["all_passed"] is True
    assert exp.first_failed_check(summary)["check"] == "all_preconditions"
    assert {row["check"] for row in summary["checks"]} == {
        "exp6745_artifact_exists",
        "exp6745_artifact_hash",
        "exp6745_corpus_ready",
        "exp6755_artifact_exists",
        "exp6755_artifact_hash",
        "exp6755_reparse_ready",
        "authentic_local_gguf_provenance",
        "exact_model_ids",
        "complete_source_rows",
        "lossless_output_bytes_and_hashes",
        "dual_encodings_and_exact_authority",
        "eligible_paired_source_cases",
        "required_model_strata",
        "required_family_strata",
    }

    missing = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        proposal_path=tmp_path / "missing.json",
    )
    assert missing["all_passed"] is False
    assert exp.first_failed_check(missing)["check"] == "exp6745_artifact_exists"

    proposal, reparse = exp.load_source_artifacts(repo_root=REPO_ROOT)
    changed = deepcopy(reparse)
    changed["rows"] = [
        row for row in changed["rows"] if row["model"]["hf_id"] != "unsloth/gemma-4-31B-it-GGUF"
    ]
    failures = exp.evaluate_source_contract(proposal, changed)
    assert set(failures["failed_checks"]) >= {
        "complete_source_rows",
        "eligible_paired_source_cases",
        "required_model_strata",
    }

    changed = deepcopy(reparse)
    changed["rows"][0]["original_output_sha256"] = "sha256:broken"
    failures = exp.evaluate_source_contract(proposal, changed)
    assert "lossless_output_bytes_and_hashes" in failures["failed_checks"]

    changed = deepcopy(reparse)
    changed["rows"][0]["encoder_b"]["exact_check"]["authority_available"] = False
    failures = exp.evaluate_source_contract(proposal, changed)
    assert "dual_encodings_and_exact_authority" in failures["failed_checks"]


def test_scenario_verify_6799_generator_preserves_outputs_and_dual_diagnostics(
    eligible_cases: list[dict], groups: list[dict]
) -> None:
    """SCENARIO-VERIFY-6799-GENERATOR keeps authentic bytes and both encodings."""

    assert len(eligible_cases) == len(groups) == exp.ELIGIBLE_CASE_COUNT == 97
    assert {case["source_model"]["hf_id"] for case in eligible_cases} == set(exp.REQUIRED_MODEL_IDS)
    assert {case["constraint_family"] for case in eligible_cases} == set(
        exp.REQUIRED_CONSTRAINT_FAMILIES
    )
    assert all(case["source_label"] == "SAT" for case in eligible_cases)
    assert all(set(group["graphs"]) == set(exp.TRANSFORMATIONS) for group in groups)

    for case in eligible_cases:
        provenance = case["source_provenance"]
        output_bytes = base64.b64decode(provenance["source_output_bytes_b64"])
        assert exp.sha256_bytes(output_bytes) == provenance["source_output_bytes_sha256"]
        assert (
            exp.sha256_bytes(provenance["source_output_envelope"].encode("utf-8"))
            == (provenance["source_output_envelope_sha256"])
        )
        assert provenance["producer_dual_encodings"]["encoder_a"] is not None
        assert provenance["producer_dual_encodings"]["encoder_b"] is not None
        assert provenance["lossless_replay_dual_encodings"]["encoder_a"]["attempted"] is True
        assert provenance["lossless_replay_dual_encodings"]["encoder_b"]["attempted"] is True
        assert provenance["translation_reasoning_diagnostic"] in exp.DIAGNOSTIC_CLASSES
        assert provenance["source_artifact_hash"] == exp.EXPECTED_SOURCE_HASHES["exp6745"]
    assert exp.LIVE_LLM_INVOKED is False


def test_scenario_verify_6799_enumeration_proves_distinct_operations_and_matching(
    groups: list[dict], rows: list[dict]
) -> None:
    """SCENARIO-VERIFY-6799-ENUMERATION checks support, topology, and nuisances."""

    assert exp.validate_probe_groups(groups) == []
    assert len(rows) == exp.ELIGIBLE_CASE_COUNT * len(exp.TRANSFORMATIONS) * 2
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert exp.audit_feature_contract(rows) == []

    for group in groups:
        base = group["graphs"]["base"]
        refinement = group["graphs"]["refinement"]
        restructuring = group["graphs"]["restructuring"]
        base_set = exp.assignment_set(base["valid_assignments"])
        refinement_set = exp.assignment_set(refinement["valid_assignments"])
        restructuring_set = exp.assignment_set(restructuring["valid_assignments"])

        assert refinement_set < base_set
        assert refinement["operation_proof"]["dependency_topology_unchanged"] is True
        assert restructuring["operation_proof"]["dependency_topology_changed"] is True
        assert base_set - restructuring_set
        assert restructuring_set - base_set
        assert restructuring["operation_proof"]["not_merely_added_constraint"] is True
        assert group["operation_class_distinction_proved"] is True

        receipt = group["matching_receipt"]
        assert receipt["all_tolerances_passed"] is True
        assert receipt["variable_count_delta"] == 0
        assert receipt["solution_count_band_match"] is True
        assert receipt["enumerated_assignment_count_delta"] == 0
        assert receipt["serialized_length_delta"] <= receipt["serialized_length_tolerance"]
        assert receipt["literal_check_work_relative_delta"] <= exp.WORK_RELATIVE_TOLERANCE
        assert receipt["difficulty_score_delta"] <= exp.DIFFICULTY_SCORE_TOLERANCE

        for transformation in exp.TRANSFORMATIONS:
            graph_record = group["graphs"][transformation]
            replay = exp.enumerate_graph(graph_record["graph"])
            assert replay["valid_assignments"] == graph_record["valid_assignments"]
            assert replay["valid_set_hash"] == graph_record["valid_set_hash"]
            assert exp.graph_hash(graph_record["graph"]) == graph_record["graph_hash"]

    group_index = {group["source_case_id"]: group for group in groups}
    for row in rows:
        group = group_index[row["source_case_id"]]
        graph = group["graphs"][row["transformation"]]["graph"]
        receipt = exp.exact_check_candidate(graph, row["candidate_assignment"])
        assert receipt == row["exact_check_receipt"]
        assert row["row_sha256"] == exp.row_checksum(row)
        if row["adversarial_condition"] == "exact_valid_witness":
            assert receipt["exact_valid"] is True
        else:
            assert row["adversarial_condition"] == "local_pass_cross_dependency_fail"
            assert receipt["local_checks_passed"] is True
            assert receipt["exact_valid"] is False
            assert receipt["failed_clause_ids"]


def test_scenario_verify_6799_splits_are_case_isolated_and_cover_all_strata(
    groups: list[dict], rows: list[dict]
) -> None:
    """SCENARIO-VERIFY-6799-SPLITS keeps each stream case in one split."""

    manifest = exp.build_split_manifest(groups, rows)
    assert set(manifest["splits"]) == {"development", "held_case"}
    assert manifest["case_overlap"] == []
    assert manifest["source_case_overlap"] == []
    assert manifest["all_required_strata_represented"] is True
    assert sum(value["source_case_count"] for value in manifest["splits"].values()) == 97

    split_by_cluster: dict[str, set[str]] = {}
    for row in rows:
        split_by_cluster.setdefault(row["case_cluster_key"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_cluster.values())
    for value in manifest["splits"].values():
        assert set(value["model_hub_ids"]) == set(exp.REQUIRED_MODEL_IDS)
        assert set(value["constraint_families"]) == set(exp.REQUIRED_CONSTRAINT_FAMILIES)


def test_scenario_verify_6799_mutations_detect_shortcuts_and_mislabels(
    groups: list[dict], rows: list[dict]
) -> None:
    """SCENARIO-VERIFY-6799-MUTATIONS rejects semantic and feature corruption."""

    attacks = exp.run_adversarial_attacks(groups, rows)
    assert set(attacks) == set(exp.ATTACK_NAMES)
    assert all(receipt["passed"] for receipt in attacks.values())
    assert attacks["solution_preserving_rename"]["semantics_preserved"] is True
    assert attacks["source_model_label_shuffle"]["feature_hashes_unchanged"] is True
    assert attacks["parser_disagreement"]["observed_diagnostic"] == ("translation_disagreement")

    duplicate = deepcopy(rows)
    duplicate.append(deepcopy(duplicate[0]))
    assert "duplicate row IDs" in exp.validate_rows(duplicate)

    leaked = deepcopy(rows[:1])
    leaked[0]["proposal_features"]["exact_valid"] = True
    assert exp.audit_feature_contract(leaked) == [f"{leaked[0]['row_id']}.exact_valid"]

    mislabeled = deepcopy(groups[:1])
    mislabeled[0]["graphs"]["restructuring"] = deepcopy(mislabeled[0]["graphs"]["refinement"])
    mislabeled[0]["graphs"]["restructuring"]["operation_class"] = "restructuring"
    assert any("restructuring" in error for error in exp.validate_probe_groups(mislabeled))

    parser_a = deepcopy(groups[0]["source_provenance"]["lossless_replay_dual_encodings"])
    parser_b = deepcopy(parser_a)
    parser_b["encoder_b"]["normalized_constraints"] = {"claim": "changed"}
    assert (
        exp.diagnose_dual_encoding(parser_b["encoder_a"], parser_b["encoder_b"], "reasoning_error")
        == "translation_disagreement"
    )


def test_scenario_verify_6799_fresh_process_replays_every_graph_hash(
    groups: list[dict], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-6799-REPLAY crosses a process boundary for exact labels."""

    replay = exp.run_exact_replay(groups, repo_root=REPO_ROOT)
    assert replay["agreement"] is True
    assert replay["fresh_process"] is True
    assert replay["replayed_graph_count"] == exp.ELIGIBLE_CASE_COUNT * 3
    assert replay["mismatches"] == []
    assert replay["cold_pid"] != replay["producer_pid"]

    direct = exp.replay_payload(groups[:1])
    assert direct["agreement"] is True
    changed = deepcopy(groups[:1])
    changed[0]["graphs"]["base"]["valid_set_hash"] = "sha256:broken"
    direct = exp.replay_payload(changed)
    assert direct["agreement"] is False
    assert direct["mismatches"] == [f"{changed[0]['source_case_id']}|base"]

    worker_payload = json.dumps({"groups": groups[:1]})
    monkeypatch.setattr(exp.sys, "stdin", StringIO(worker_payload))
    assert exp._exact_replay_worker() == 0
    worker_receipt = json.loads(capsys.readouterr().out)
    assert worker_receipt["agreement"] is True
    assert worker_receipt["replayed_graph_count"] == 3

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9, stdout="", stderr="forced"),
    )
    with pytest.raises(RuntimeError, match="forced"):
        exp.run_exact_replay(groups[:1], repo_root=REPO_ROOT)


def test_req_verify_6799_artifact_is_complete_ready_and_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-6799 writes the required ready artifact without live inference."""

    output_path = tmp_path / "fixture.json"
    artifact = exp.write_outputs(
        run_date="20260831",
        artifact_path=output_path,
        repo_root=REPO_ROOT,
        duration_s=2.5,
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["live_llm_invoked"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_output_constraint_probe_ready"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert {row["hf_id"] for row in artifact["source_model_specs"]} == set(exp.REQUIRED_MODEL_IDS)
    assert len(artifact["graph_hashes"]) == exp.ELIGIBLE_CASE_COUNT * 3
    assert len(artifact["exact_replay_receipts"]) == exp.ELIGIBLE_CASE_COUNT * 3
    assert all(row["fresh_process"] for row in artifact["exact_replay_receipts"])
    assert artifact["gate_check_summary"]["all_passed"] is True

    changed = deepcopy(artifact)
    changed["rows"].append(deepcopy(changed["rows"][0]))
    assert "duplicate row IDs" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"][0]["proposal_features"]["source_model_hf_id"] = "leak"
    assert any("source_model_hf_id" in error for error in exp.validate_artifact(changed))
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "bad"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["live_llm_invoked"] = True
    assert "live_llm_invoked must remain false" in exp.validate_artifact(changed)


def test_scenario_verify_6799_blocked_artifact_stops_without_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6799-BLOCKED writes full evidence and no substitute rows."""

    artifact = exp.build_artifact(
        run_date="20260831",
        repo_root=REPO_ROOT,
        proposal_path=tmp_path / "missing.json",
        duration_s=0.1,
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_blocked_model_output_probe_fixture"
    assert artifact["rows"] == []
    assert artifact["graph_hashes"] == []
    assert artifact["model_output_constraint_probe_ready"] is False
    assert artifact["live_llm_invoked"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_model_output_probe_fixture")
    assert artifact["gate_check_summary"]["first_failure"]["check"] == ("exp6745_artifact_exists")

    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(run_date="2026-08-31", repo_root=REPO_ROOT)

    relative = exp.write_outputs(
        run_date="20260831",
        artifact_path=Path("blocked.json"),
        repo_root=tmp_path,
        proposal_path=tmp_path / "missing.json",
        duration_s=0.1,
    )
    assert relative["model_output_constraint_probe_ready"] is False
    assert (tmp_path / "blocked.json").is_file()


def test_req_verify_6799_cli_writes_only_the_requested_output(tmp_path: Path) -> None:
    """REQ-VERIFY-6799 exposes the mandated date command with an isolated output."""

    output_path = tmp_path / "cli.json"
    assert exp.main(["--date", "20260831", "--output", str(output_path)]) == 0
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["model_output_constraint_probe_ready"] is True
    assert artifact["duration_s"] >= 0.0001
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6799_defensive_mutations_cover_rejection_paths(
    sources: tuple[dict, dict],
    eligible_cases: list[dict],
    groups: list[dict],
    rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6799 rejects malformed evidence at every public boundary."""

    proposal, reparse = sources
    assert exp._decoded_legacy_output("not a literal") is None
    assert exp._decoded_legacy_output("'plain text'") is None
    assert exp._solution_count_band(0) == "0"
    assert exp.diagnose_dual_encoding({}, {}, "reasoning_error") == "translation_failure"
    parser_error = {"attempted": True, "error": "broken"}
    assert (
        exp.diagnose_dual_encoding(parser_error, parser_error, "reasoning_error")
        == "translation_failure"
    )
    parser_ok = {
        "attempted": True,
        "error": None,
        "normalized_constraints": {"claim": "ABSTAIN"},
    }
    assert exp.diagnose_dual_encoding(parser_ok, parser_ok, "abstention") == ("model_abstention")

    changed = deepcopy(proposal)
    changed["models_used"] = []
    assert exp._model_provenance_valid(changed) is False
    changed = deepcopy(proposal)
    changed["models_used"][0]["model_path"] = "/missing/model.gguf"
    assert exp._model_provenance_valid(changed) is False
    changed = deepcopy(proposal)
    changed["gpu_receipts"] = []
    assert exp._model_provenance_valid(changed) is False

    with monkeypatch.context() as patch:
        patch.setattr(exp, "enumerate_cnf", lambda *args: {"solution_count": 1})
        assert exp.select_eligible_cases(proposal, reparse) == []

    case = eligible_cases[0]
    base = groups[0]["graphs"]["base"]
    refinement = groups[0]["graphs"]["refinement"]
    single_index = next(
        index
        for index, group in enumerate(groups)
        if "dependency_block_size" not in group["graphs"]["restructuring"]["operation_proof"]
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            exp,
            "_matching_receipt",
            lambda *args: {"all_tolerances_passed": False},
        )
        assert (
            exp._find_restructuring(
                eligible_cases[single_index],
                groups[single_index]["graphs"]["base"],
                groups[single_index]["graphs"]["refinement"],
            )
            is None
        )
    with monkeypatch.context() as patch:
        patch.setattr(exp, "_candidate_clauses", lambda *args: ((1,),))
        patch.setattr(exp, "_clause_mask", lambda *args: 0)
        assert exp._find_block_restructuring(case, base, refinement) is None
    with monkeypatch.context() as patch:
        patch.setattr(exp, "_find_restructuring", lambda *args: None)
        patch.setattr(exp, "_find_block_restructuring", lambda *args: None)
        with pytest.raises(ValueError, match="no matched restructuring"):
            exp._build_group(case)
    unary = exp._graph_record("base", exp._make_graph(1, [[1]]), {})
    with pytest.raises(ValueError, match="no local-pass cross-dependency"):
        exp._hard_negative(unary)

    changed_groups = deepcopy(groups[:1]) * 2
    assert "duplicate source case IDs" in exp.validate_probe_groups(changed_groups)
    changed_groups = deepcopy(groups[:1])
    changed_groups[0]["graphs"].pop("base")
    assert any(
        "missing transformation graphs" in error
        for error in exp.validate_probe_groups(changed_groups)
    )
    changed_groups = deepcopy(groups[:1])
    changed_groups[0]["graphs"]["base"]["graph_hash"] = "sha256:broken"
    assert any(
        "exact evidence mismatch" in error for error in exp.validate_probe_groups(changed_groups)
    )
    changed_groups = deepcopy(groups[:1])
    changed_groups[0]["graphs"]["refinement"]["operation_class"] = "base"
    assert any(
        "invalid refinement operation" in error
        for error in exp.validate_probe_groups(changed_groups)
    )

    bad_rows = deepcopy(rows[:2])
    bad_rows[0]["exact_check_receipt"]["exact_valid"] = False
    bad_rows[1]["exact_check_receipt"] = {
        "local_checks_passed": False,
        "exact_valid": True,
        "failed_clause_ids": [],
    }
    row_errors = exp.validate_rows(bad_rows)
    assert any("invalid witness label" in error for error in row_errors)
    assert any("invalid hard negative" in error for error in row_errors)

    artifact = exp.build_artifact(run_date="20260831", repo_root=REPO_ROOT)
    changed_artifact = deepcopy(artifact)
    changed_artifact.pop("schema")
    changed_artifact["verifier_is_oracle"] = True
    changed_artifact["verdict_class"] = "unknown"
    changed_artifact["honest_verdict"] = "unfinished"
    changed_artifact["gate_check_summary"]["all_passed"] = False
    artifact_errors = exp.validate_artifact(changed_artifact)
    assert any("missing required fields" in error for error in artifact_errors)
    assert "field_principles must cover every top-level field" in artifact_errors
    assert "verifier_is_oracle must remain false" in artifact_errors
    assert "verdict_class is outside the closed enum" in artifact_errors
    assert "honest_verdict lacks a terminal prefix" in artifact_errors
    assert "ready artifact has failed gates" in artifact_errors

    blocked = deepcopy(artifact)
    blocked["model_output_constraint_probe_ready"] = False
    blocked["status"] = "wrong"
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked artifact has wrong status" in blocked_errors
    assert "blocked artifact must not contain probe rows" in blocked_errors

    monkeypatch.setattr(exp.sys, "stdin", StringIO(json.dumps({"groups": groups[:1]})))
    assert exp.main(["--exact-replay-worker"]) == 0
    assert json.loads(capsys.readouterr().out)["agreement"] is True
