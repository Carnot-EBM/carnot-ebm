"""Focused contract tests for the independently requalified span fixture.

Spec refs: REQ-VERIFY-7222 and SCENARIO-VERIFY-7222-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7222_v636_span_fixture as exp


REPO = Path(__file__).resolve().parents[2]


def test_req_verify_7222_uses_exact_cpu_identity_without_model_work() -> None:
    """REQ-VERIFY-7222 uses the registered CPU literal in both fields."""

    assert exp.INFERENCE_SUBSTRATE == "cpu_exact_solver_or_simulator"
    assert exp.INFERENCE_SUBSTRATE_CLASS == "cpu_exact_solver_or_simulator"
    assert exp.MODEL_SPECS == []
    assert exp._unwrap({"principle": "why", "value": 1}) == 1
    arbitrary = {"value": 1, "other": "not an annotation"}
    assert exp._unwrap(arbitrary) is arbitrary


def test_scenario_verify_7222_reconstruction_preserves_blind_v635_design() -> None:
    """SCENARIO-VERIFY-7222-RECONSTRUCTION keeps the frozen panel and public view."""

    panel = exp.build_panel()

    assert len(panel["public_rows"]) == 320
    assert len(panel["authority_rows"]) == 320
    assert panel["split_manifest"]["split_base_counts"] == {
        "canary": 8,
        "development": 8,
        "test": 64,
    }
    assert panel["split_manifest"]["split_hashes_disjoint"] is True
    assert all(set(row) == {"unit_id", "source_text", "claim_text"} for row in panel["public_rows"])
    assert {row["variant"] for row in panel["authority_rows"]} == {
        "supported",
        "reversal",
        "joint_support",
        "support_removed",
    }
    assert {row["expected_decision"] for row in panel["authority_rows"]} == {
        "supported",
        "contradicted",
        "unknown",
    }


def test_scenario_verify_7222_execution_uses_compiler_executor_and_private_labels() -> None:
    """SCENARIO-VERIFY-7222-EXECUTION checks all semantic units independently."""

    panel = exp.build_panel()
    rows = exp.execute_panel(panel["public_rows"], panel["authority_rows"])

    assert len(rows) == 320
    assert sum(row["split"] == "test" for row in rows) == 256
    assert all(row["metric"] == 1 and row["error"] is None for row in rows)
    assert {row["prediction"] for row in rows} == {"supported", "contradicted", "unknown"}
    assert exp.AUTHORITY_IMPORTS_CANDIDATE is False


def test_scenario_verify_7222_mutations_fail_relevant_boundaries() -> None:
    """SCENARIO-VERIFY-7222-MUTATIONS catches span, direction, and premise attacks."""

    rows = exp.mutation_rows()
    by_name = {row["mutation"]: row for row in rows}

    assert {
        "out_of_range",
        "cross_document",
        "wrong_sentence",
        "type_mismatch",
        "unsupported_predicate",
        "invalid_polarity",
        "excess_source_relations",
        "inverse_relation",
        "direction_reversal",
        "negation",
        "support_removed",
        "grammar_serialization",
    } <= set(by_name)
    assert all(row["passed"] is True for row in rows)
    assert by_name["out_of_range"]["check_failed_as_required"] is True
    assert by_name["direction_reversal"]["check_failed_as_required"] is True
    assert by_name["support_removed"]["check_failed_as_required"] is True


def test_scenario_verify_7222_artifact_builds_fresh_raw_files_and_clean_receipt(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7222-ARTIFACT seals and verifies a new candidate checkpoint."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    public_path = tmp_path / exp.PUBLIC_VIEW_PATH
    authority_path = tmp_path / exp.AUTHORITY_SIDECAR_PATH
    manifest_path = tmp_path / exp.FIXTURE_MANIFEST_PATH
    checkpoint_path = tmp_path / exp.CHECKPOINT_PATH

    assert artifact["status"] == "complete"
    assert artifact["span_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["inference_substrate_class"] == exp.INFERENCE_SUBSTRATE_CLASS
    assert artifact["execution_venue"] == "host" and artifact["execution_host"]
    assert all(path.is_file() for path in (public_path, authority_path, manifest_path))
    assert checkpoint_path.is_file()
    assert exp.validate_artifact(artifact, REPO, output_root=tmp_path) == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["paths"] == {
        "public_view": exp.PUBLIC_VIEW_PATH.as_posix(),
        "authority_sidecar": exp.AUTHORITY_SIDECAR_PATH.as_posix(),
        "fixture_manifest": exp.FIXTURE_MANIFEST_PATH.as_posix(),
    }
    assert manifest["counts"]["public_rows"] == 320
    assert manifest["counts"]["authority_rows"] == 320
    receipt = artifact["substrate_classifier_receipt"]
    assert receipt["historical_exp7208"]["stored_flagged_adversarial"] is True
    assert receipt["historical_exp7208"]["verifier_report"]["flag_count"] >= 1
    assert receipt["candidate_checkpoint"]["duration_floor"]["min_duration_s"] == 0.0001
    assert receipt["candidate_checkpoint"]["verifier_report"]["flag_count"] == 0


def test_scenario_verify_7222_artifact_fails_closed_on_tampering(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7222-ARTIFACT rejects forged rows, hashes, and readiness."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)

    changed = deepcopy(artifact)
    changed["rows"][0]["metric"] = 0
    assert "rows" in exp.validate_artifact(changed, REPO, output_root=tmp_path)

    changed = deepcopy(artifact)
    changed["substrate_classifier_receipt"]["candidate_checkpoint"]["verifier_report"][
        "flag_count"
    ] = 1
    assert "substrate_classifier_receipt" in exp.validate_artifact(
        changed, REPO, output_root=tmp_path
    )


def test_scenario_verify_7222_preflight_blocks_missing_authenticated_input(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7222-PREFLIGHT names the exact missing upstream file."""

    missing = tmp_path / "missing-exp7196.json"
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_root=tmp_path / "outputs",
        path_overrides={"exp7196_artifact": missing},
    )

    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["span_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "source_exists",
        "upstream": "exp7196_artifact",
        "field": "path",
        "expected_value": "existing_file",
        "observed_value": str(missing),
    }
    assert exp.validate_artifact(artifact, REPO, output_root=tmp_path / "outputs") == []


def test_req_verify_7222_checksum_ignores_only_process_local_timing() -> None:
    """REQ-VERIFY-7222 binds evidence while allowing measured timing to vary."""

    left = {
        "duration_s": 1.0,
        "timestamps": {"started_at_utc": "a"},
        "reproducibility_checksum": "old",
        "rows": [1],
    }
    right = {
        "duration_s": 2.0,
        "timestamps": {"started_at_utc": "b"},
        "reproducibility_checksum": "new",
        "rows": [1],
    }
    assert exp.artifact_checksum(left) == exp.artifact_checksum(right)
    right["rows"] = [2]
    assert exp.artifact_checksum(left) != exp.artifact_checksum(right)


def test_req_verify_7222_date_and_main_contract(monkeypatch, tmp_path: Path) -> None:
    """REQ-VERIFY-7222 accepts only the fixed date and exits after a valid artifact."""

    monkeypatch.setattr(exp, "find_repo_root", lambda start: REPO)
    monkeypatch.setattr(
        exp,
        "build_artifact",
        lambda root, run_date: {
            "honest_verdict": "complete_circular_positive_span_fixture_ready",
            "span_fixture_ready_score": 1,
        },
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: [])

    assert exp.main(["--date", exp.RUN_DATE]) == 0


def test_scenario_verify_7222_validator_reports_each_terminal_contract(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7222-ARTIFACT makes every required terminal field falsifiable."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    cases = (
        ("field_principles", None, "field_principles"),
        ("run_date", "bad", "run_date"),
        ("execution_host", "", "execution_identity"),
        ("MODEL_SPECS", [{"not": "invoked"}], "model_contract"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("duration_s", True, "duration_s"),
        ("random_seed", 1, "random_seed"),
        ("inference_substrate", "free_text", "inference_substrate"),
        ("inference_substrate_class", "aggregation", "inference_substrate_class"),
        ("verdict_class", "positive", "readiness_terminal_state"),
        ("sample_size_budget", {}, "sample_size_budget"),
        ("split_manifest", {}, "split_manifest"),
        ("lexical_control_rows", [], "lexical_control_rows"),
        ("mutation_rows", [], "mutation_rows"),
        ("grammar_contract", {}, "grammar_contract"),
        ("source_artifact_hashes", {}, "source_artifact_hashes"),
        ("substrate_classifier_receipt", [], "substrate_classifier_receipt"),
        ("readiness_checks", {}, "readiness_checks"),
    )
    for field, value, expected in cases:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, REPO, output_root=tmp_path)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "forged"
    assert "reproducibility_checksum" in exp.validate_artifact(changed, REPO, output_root=tmp_path)
    assert exp.validate_artifact([]) == ["artifact_mapping"]
    assert exp.validate_artifact({})[0].startswith("missing_required_field:")


def test_scenario_verify_7222_validator_rejects_changed_raw_files(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7222-ARTIFACT checks each sealed public, private, and manifest byte."""

    artifact = exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path)
    public_path = tmp_path / exp.PUBLIC_VIEW_PATH
    authority_path = tmp_path / exp.AUTHORITY_SIDECAR_PATH
    manifest_path = tmp_path / exp.FIXTURE_MANIFEST_PATH
    originals = {path: path.read_bytes() for path in (public_path, authority_path, manifest_path)}

    public_rows = [json.loads(line) for line in originals[public_path].decode().splitlines()]
    public_rows[0]["source_text"] = "Aster precedes Brin."
    public_path.write_text("\n".join(exp.canonical_json(row) for row in public_rows) + "\n")
    assert "public_view" in exp.validate_artifact(artifact, REPO, output_root=tmp_path)
    public_path.write_bytes(originals[public_path])

    authority_rows = [json.loads(line) for line in originals[authority_path].decode().splitlines()]
    authority_rows[0]["expected_decision"] = "unknown"
    authority_path.write_text("\n".join(exp.canonical_json(row) for row in authority_rows) + "\n")
    assert "authority_sidecar" in exp.validate_artifact(artifact, REPO, output_root=tmp_path)
    authority_path.write_bytes(originals[authority_path])

    manifest = json.loads(originals[manifest_path])
    manifest["counts"]["public_rows"] = 319
    manifest_path.write_text(json.dumps(manifest))
    assert "fixture_manifest" in exp.validate_artifact(artifact, REPO, output_root=tmp_path)
    manifest_path.write_bytes(originals[manifest_path])

    public_path.unlink()
    assert "sealed_fixture_files" in exp.validate_artifact(artifact, REPO, output_root=tmp_path)
    public_path.write_bytes(originals[public_path])


def test_scenario_verify_7222_blocked_and_nonterminal_validation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7222-PREFLIGHT distinguishes a valid block from forged state."""

    missing = tmp_path / "missing.json"
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_root=tmp_path / "outputs",
        path_overrides={"exp7196_artifact": missing},
    )
    changed = deepcopy(artifact)
    changed["span_fixture_ready_score"] = 1
    changed["gate_check_summary"] = {"passed": True}
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    errors = exp.validate_artifact(changed, REPO, output_root=tmp_path / "outputs")
    assert {"blocked_terminal_state", "gate_check_summary"} <= set(errors)

    running = exp._base_artifact(exp.RUN_DATE)
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "status" in exp.validate_artifact(running)


def _stub_base_preconditions(monkeypatch) -> None:
    """Let one test target the task-owned checks after authenticated producer checks."""

    monkeypatch.setattr(
        exp.v635,
        "_preconditions",
        lambda root, run_date, output_root, overrides: ([{"passed": True}], {}, {}),
    )


def test_scenario_verify_7222_task_owned_precondition_failures(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7222-PREFLIGHT covers missing, malformed, stale, and unwritable inputs."""

    _stub_base_preconditions(monkeypatch)
    missing = tmp_path / "missing-module.py"
    checks, _, _ = exp._preconditions(REPO, exp.RUN_DATE, tmp_path / "missing", {"module": missing})
    assert checks[-1]["check"] == "source_exists" and checks[-1]["passed"] is False

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "malformed", {"exp7208_artifact": malformed}
    )
    assert checks[-1]["check"] == "source_parse" and checks[-1]["passed"] is False

    no_requirement = tmp_path / "spec.md"
    no_requirement.write_text("no driving requirement")
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "spec", {"constraint_spec": no_requirement}
    )
    assert checks[-1]["check"] == "driving_spec" and checks[-1]["passed"] is False

    historical = json.loads((REPO / exp.EXP7208_PATH).read_text())
    historical["flagged_adversarial"] = False
    unflagged = tmp_path / "unflagged.json"
    unflagged.write_text(json.dumps(historical))
    checks, _, _ = exp._preconditions(
        REPO, exp.RUN_DATE, tmp_path / "unflagged", {"exp7208_artifact": unflagged}
    )
    assert checks[-1]["check"] == "historical_quarantine_preserved"
    assert checks[-1]["passed"] is False

    real_access = os.access
    monkeypatch.setattr(
        exp.os,
        "access",
        lambda path, mode: False if mode == os.W_OK else real_access(path, mode),
    )
    checks, _, _ = exp._preconditions(REPO, exp.RUN_DATE, tmp_path / "unwritable", None)
    assert checks[-1]["check"] == "output_destination" and checks[-1]["passed"] is False


def test_req_verify_7222_jsonl_and_date_reject_bad_values(tmp_path: Path) -> None:
    """REQ-VERIFY-7222 rejects non-object raw rows and an unfrozen execution date."""

    path = tmp_path / "rows.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="JSONL row"):
        exp._read_jsonl(path)
    with pytest.raises(Exception, match="run date"):
        exp._date_argument("20260910")


def test_scenario_verify_7222_dirty_candidate_and_invalid_block_stop(
    monkeypatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-7222-ARTIFACT never promotes a dirty candidate or invalid block."""

    original_verify = exp.adversarial_verify.verify_artifact

    def dirty_verify(path, declared=False):
        if "7208" in Path(path).name:
            return original_verify(path, declared=declared)
        return {"artifact": str(path), "loaded": True, "flag_count": 1, "flags": [{}]}

    monkeypatch.setattr(exp.adversarial_verify, "verify_artifact", dirty_verify)
    with pytest.raises(ValueError, match="invalid Exp7222 artifact"):
        exp.build_artifact(REPO, exp.RUN_DATE, output_root=tmp_path / "dirty")

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid blocked Exp7222 artifact"):
        exp.build_artifact(
            REPO,
            exp.RUN_DATE,
            output_root=tmp_path / "blocked",
            path_overrides={"exp7196_artifact": tmp_path / "absent.json"},
        )


def test_req_verify_7222_main_returns_nonzero_for_invalid_terminal(monkeypatch) -> None:
    """REQ-VERIFY-7222 makes the CLI fail when final cold validation disagrees."""

    monkeypatch.setattr(exp, "find_repo_root", lambda start: REPO)
    monkeypatch.setattr(exp, "build_artifact", lambda root, run_date: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: ["forced"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
