"""Tests for the sealed memory pathway portability audit.

Spec refs: REQ-CL-6842, SCENARIO-CL-6842-PRECONDITIONS,
SCENARIO-CL-6842-FRESH-REDUCTION, SCENARIO-CL-6842-SHARD-IDENTITY,
SCENARIO-CL-6842-ATTACKS, SCENARIO-CL-6842-DURABILITY,
SCENARIO-CL-6842-PORTABILITY, and SCENARIO-CL-6842-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6842_sealed_memory_pathway_portability_audit as exp
import scripts.adversarial_verify as adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = exp.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """SCENARIO-CL-6842-PRECONDITIONS loads the two sealed shard artifacts."""

    return exp.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-CL-6842 builds one deterministic sealed reducer artifact."""

    return exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=tmp_path_factory.mktemp("exp6842-state"),
        run_date="20260901",
        duration_s=0.25,
    )


def test_req_cl_6842_spec_precedes_implementation() -> None:
    """REQ-CL-6842 declares the reducer paths, fields, and scenarios."""

    spec = (REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6842", 1)[1]
    for requirement_id in exp.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (exp.MODULE_RELATIVE_PATH, exp.SCRIPT_RELATIVE_PATH, exp.RESULT_RELATIVE_PATH):
        assert path.as_posix() in section


def test_scenario_cl_6842_preconditions_accept_sources(sources: dict[str, dict]) -> None:
    """SCENARIO-CL-6842-PRECONDITIONS accepts complete shard inputs."""

    summary = exp.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("a_score", "csl_shard_a_complete_score"),
        ("b_score", "csl_shard_b_complete_score"),
        ("source_hash", "source_artifact_hashes"),
        ("overlap", "disjoint_source_identities"),
        ("complete_rows", "complete_arm_rows"),
        ("receipt", "exact_outcome_receipts"),
    ],
)
def test_scenario_cl_6842_preconditions_fail_closed(
    sources: dict[str, dict],
    tmp_path: Path,
    fault: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6842-PRECONDITIONS records blocked gate values."""

    changed = deepcopy(sources)
    paths = dict(SOURCE_PATHS)
    if fault == "a_score":
        changed["exp6840"]["csl_shard_a_complete_score"] = 0.0
    elif fault == "b_score":
        changed["exp6841"]["csl_shard_b_complete_score"] = 0.0
    elif fault == "source_hash":
        paths["exp6841"] = tmp_path / "missing.json"
    elif fault == "overlap":
        overlap_id = changed["exp6840"]["rows"][0]["source_event_row_id"]
        changed["exp6841"]["rows"][0]["source_event_row_id"] = overlap_id
    elif fault == "complete_rows":
        changed["exp6840"]["rows"] = changed["exp6840"]["rows"][1:]
    else:
        changed["exp6841"]["rows"][0]["exact_outcome"]["exact_outcome_hash"] = None

    summary = exp.check_preconditions(changed, paths)
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert summary["passed"] is False
    assert failed["passed"] is False
    assert "observed" in failed

    blocked = exp.build_artifact(
        changed,
        source_paths=paths,
        state_root=tmp_path,
        run_date="20260901",
        duration_s=0.25,
    )
    assert blocked["status"] == exp.BLOCKED_STATUS
    assert blocked["rows"] == []
    assert blocked["sealed_csl_audit_complete_score"] == 0.0
    assert blocked["continuous_self_learning_ready_score"] == 0.0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith(exp.BLOCKED_STATUS)
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]

    receiptless = deepcopy(sources)
    receiptless["exp6840"]["rows"][0]["exact_outcome"] = None
    bad_receipt = exp.check_preconditions(receiptless, SOURCE_PATHS)
    assert "exact_outcome_receipts" in bad_receipt["failed_checks"]

    malformed_rows = {"exp6840": {"rows": "not-a-list"}, "exp6841": {"rows": []}}
    assert exp.expected_source_row_count(malformed_rows) == 0


def test_scenario_cl_6842_fresh_reduction_recomputes_all_effects(artifact: dict) -> None:
    """SCENARIO-CL-6842-FRESH-REDUCTION keeps losses visible by split."""

    fresh_rows = [row for row in artifact["rows"] if row["attack"] == exp.FRESH_ATTACK]
    assert len(fresh_rows) == exp.expected_source_row_count(artifact)
    assert all(exp.REQUIRED_ROW_FIELDS <= set(row) for row in fresh_rows)
    assert all(row["matched_no_memory_baseline"] for row in fresh_rows)

    no_memory = artifact["fresh_reduction_results"][exp.NO_MEMORY_ARM]
    assert no_memory["wins"] == 0
    assert no_memory["losses"] == 0
    assert no_memory["ties"] == no_memory["held_future_rows"]

    verified = artifact["fresh_reduction_results"][exp.VERIFIED_RESIDUAL_ARM]
    assert verified["held_future_rows"] > 0
    assert verified["losses"] > 0
    assert verified["mean_effect_vs_no_memory"] < 0.0
    assert any(
        group["losses"] > group["wins"]
        for group in verified["by_family_order_seed"].values()
        if group["held_future_rows"] > 0
    )
    assert all("uncertainty" in group for group in verified["by_family_order_seed"].values())


def test_scenario_cl_6842_shard_identity_is_disjoint_and_row_local(
    sources: dict[str, dict],
    artifact: dict,
) -> None:
    """SCENARIO-CL-6842-SHARD-IDENTITY preserves independent shard rows."""

    identity = exp.source_identity_summary(sources)
    assert identity["disjoint_source_event_identities"] is True
    assert identity["exp6840_unique_source_events"] == 459
    assert identity["exp6841_unique_source_events"] == 306
    assert artifact["source_identity_summary"] == identity
    assert set(artifact["source_artifact_hashes"]) == {"exp6840", "exp6841"}
    assert not any(row.get("producer_aggregate_imported") for row in artifact["rows"])


def test_scenario_cl_6842_attacks_are_complete_and_bounded(artifact: dict) -> None:
    """SCENARIO-CL-6842-ATTACKS covers all edit and pathway interventions."""

    row_attacks = {row["attack"] for row in artifact["rows"]}
    assert set(exp.ATTACK_NAMES) <= row_attacks

    deletion = artifact["deletion_results"]
    assert deletion["rows"] == exp.expected_source_row_count(artifact)
    assert deletion["deleted_memory_action_count"] > 0
    assert deletion["readiness_gate_passed"] is False

    substitution = artifact["substitution_results"]
    assert substitution["substituted_action_count"] > 0
    assert substitution["readiness_gate_passed"] is False

    reorder = artifact["reorder_results"]
    assert reorder["order_identity_changed"] is True
    assert reorder["row_count_preserved"] is True
    assert reorder["effect_count_preserved"] is True

    poison = artifact["poison_results"]
    assert poison["bounded_poison_cases"] > 0
    assert poison["poison_accepted_count"] == 0
    assert poison["readiness_gate_passed"] is False

    stale = artifact["stale_credit_results"]
    assert stale["stale_credit_cases"] > 0
    assert stale["stale_credit_accepted_count"] == 0
    assert stale["readiness_gate_passed"] is False

    pathways = artifact["latent_error_pathways"]
    assert {"persisted", "repaired", "expired", "rolled_back"} <= set(pathways["state_counts"])
    committed_only = {
        "memory_transition_statuses": ["committed"],
        "arm": exp.VERIFIED_RESIDUAL_ARM,
    }
    assert exp._pathway_state(committed_only) == "persisted"
    assert exp._uncertainty([1.0]) == 0.0

    capacity = artifact["capacity_results"]
    assert capacity["capacity_attack_complete"] is True
    assert capacity["max_active_count"] <= capacity["capacity_budget"]
    assert capacity["overflow_commit_count"] == 0


def test_scenario_cl_6842_durability_restart_and_rollback_are_exact(
    artifact: dict,
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6842-DURABILITY preserves restart and rollback hashes."""

    source_rows = [row for row in artifact["rows"] if row["attack"] == exp.FRESH_ATTACK][:20]
    receipt = exp.restart_and_rollback_checks(source_rows, state_root=tmp_path / "state")
    assert receipt["restart"]["matches_clean_replay"] is True
    assert receipt["restart"]["loaded_state_hash"] == receipt["restart"]["clean_replay_state_hash"]
    assert receipt["rollback"]["restored_parent_hash"] is True
    assert receipt["capacity"]["overflow_commit_count"] == 0

    assert artifact["restart_durability_results"]["matches_clean_replay"] is True
    assert artifact["rollback_results"]["restored_parent_hash"] is True


def test_scenario_cl_6842_portability_and_ready_gate_are_conjunctive(artifact: dict) -> None:
    """SCENARIO-CL-6842-PORTABILITY and READY keep null evidence terminal."""

    leave_one = artifact["leave_one_family_out_results"]
    families = artifact["fresh_reduction_results"][exp.VERIFIED_RESIDUAL_ARM]["families"]
    assert set(leave_one) == set(families)
    assert all(result["held_future_rows"] > 0 for result in leave_one.values())
    assert any(result["portability_gate_passed"] is False for result in leave_one.values())

    negative = artifact["negative_transfer_results"][exp.VERIFIED_RESIDUAL_ARM]
    assert negative["negative_transfer_count"] > 0
    assert negative["readiness_gate_passed"] is False

    dose = artifact["action_dose_calibration"]
    assert dose["bounded_dose_gate_passed"] is True
    assert dose["calibrated_dose_gate_passed"] is False
    assert artifact["sealed_csl_audit_complete_score"] == 1.0
    assert artifact["continuous_self_learning_ready_score"] == 0.0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")


def test_scenario_cl_6842_artifact_validation_and_cli(
    artifact: dict,
    sources: dict[str, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6842 validates fields, checksums, writer, CLI, and wrapper."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []
    assert exp.expected_source_row_count(
        {"rows": artifact["rows"]}
    ) == exp.expected_source_row_count(artifact)

    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    assert floor is not None
    assert floor["reason"] == "deterministic_verifier"

    changed = dict(artifact)
    changed.pop("rows")
    assert "required field set mismatch" in exp.validate_artifact(changed)
    changed = {**artifact, "field_principles": dict(artifact["field_principles"])}
    changed["field_principles"].pop("rows")
    assert "field_principles coverage mismatch" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["continuous_self_learning_task"] = False
    assert "continuous_self_learning_task must be true" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["honest_verdict"] = "blocked: no terminal prefix"
    assert "honest_verdict lacks complete_ prefix" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["sealed_csl_audit_complete_score"] = 0.0
    assert "complete artifact missing audit score" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["continuous_self_learning_ready_score"] = 2.0
    assert "ready score outside binary gate" in exp.validate_artifact(changed)
    changed = {**artifact, "continuous_self_learning_ready_score": 1.0}
    assert "ready score passed despite failed readiness gate" in exp.validate_artifact(changed)
    changed = {**artifact, "rows": artifact["rows"][:-1]}
    assert "complete row count mismatch" in exp.validate_artifact(changed)
    first_row = dict(artifact["rows"][0])
    first_row.pop("row_id")
    changed = {**artifact, "rows": [first_row, *artifact["rows"][1:]]}
    assert "row field coverage mismatch" in exp.validate_artifact(changed)
    changed = dict(artifact)
    changed["status"] = exp.BLOCKED_STATUS
    assert "blocked artifact must not expose rows" in exp.validate_artifact(changed)

    result_path = tmp_path / "artifact.json"
    exp.write_artifact(result_path, artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--result-path", str(result_path)]) == 0

    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.write_artifact(tmp_path / "bad.json", {"field_principles": {}})

    bad_validate_path = tmp_path / "bad-validate.json"
    bad_validate_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
        exp.main(["--validate", "--result-path", str(bad_validate_path)])

    load_list = tmp_path / "list.json"
    load_bad = tmp_path / "bad-source.json"
    load_list.write_text("[]", encoding="utf-8")
    load_bad.write_text("{", encoding="utf-8")
    loaded = exp.load_sources(
        {"list": load_list, "bad": load_bad, "missing": tmp_path / "missing.json"}
    )
    assert loaded["list"] == {"_load_error": "not_object"}
    assert loaded["bad"] == {"_load_error": "JSONDecodeError"}
    assert loaded["missing"] == {"_load_error": "FileNotFoundError"}

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced build error"])
    with pytest.raises(ValueError, match="forced build error"):
        exp.build_artifact(
            sources,
            source_paths=SOURCE_PATHS,
            state_root=tmp_path / "forced-build",
            run_date="20260901",
            duration_s=0.25,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validate)

    temp_artifact = exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=None,
        run_date="20260901",
        duration_s=0.25,
    )
    assert temp_artifact["restart_durability_results"]["matches_clean_replay"] is True

    generated_path = tmp_path / "generated.json"
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: artifact)
    assert exp.main(["--date", "20260901", "--result-path", str(generated_path)]) == 0
    assert json.loads(generated_path.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
        exp.main(["--date", "20260901", "--result-path", str(tmp_path / "bad-main.json")])

    script = REPO_ROOT / exp.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(
        "sys.argv",
        [script.name, "--validate", "--result-path", str(result_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
