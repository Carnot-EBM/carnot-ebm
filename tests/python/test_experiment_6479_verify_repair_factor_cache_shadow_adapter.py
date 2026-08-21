"""Tests for Exp6479 verify-repair factor-cache shadow adapter.

Spec refs: REQ-PIPELINE-6479, SCENARIO-PIPELINE-6479-SHADOW,
REQ-LEARN-6479, SCENARIO-LEARN-6479-EXACT-ADMIT,
SCENARIO-LEARN-6479-RESTART, SCENARIO-LEARN-6479-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_6479_verify_repair_factor_cache_shadow_adapter as mod


REPO = Path(__file__).resolve().parents[2]


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.run(
        date="20260821",
        root=REPO,
        result_path=tmp_path / "experiment_6479.json",
        ledger_dir=tmp_path / "ledger",
        duration_s=1.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=write,
    )


def test_req_learn_6479_specs_and_required_fields_are_declared() -> None:
    """REQ-LEARN-6479: specs name the production and artifact contract."""

    pipeline_spec = (REPO / "openspec/capabilities/pipeline/spec.md").read_text(encoding="utf-8")
    learning_spec = (REPO / "openspec/capabilities/continuous-learning/spec.md").read_text(
        encoding="utf-8"
    )
    combined = pipeline_spec + "\n" + learning_spec

    for marker in (
        "REQ-PIPELINE-6479",
        "SCENARIO-PIPELINE-6479-SHADOW",
        "REQ-LEARN-6479",
        "SCENARIO-LEARN-6479-EXACT-ADMIT",
        "SCENARIO-LEARN-6479-RESTART",
        "SCENARIO-LEARN-6479-ARTIFACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "default-off",
        "exact checker",
    ):
        assert marker in combined

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in mod.REQUIRED_ARTIFACT_FIELDS
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_6479_artifact_rows_gates_and_ready_score(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6479-ARTIFACT: row reductions drive readiness."""

    artifact = _artifact(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "success_ready"
    assert artifact["inference_substrate"] == "deterministic_pipeline_integration_no_llm"
    assert artifact["verifier_is_oracle"]["value"] is True
    assert artifact["factor_cache_shadow_adapter_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["adapter_api_and_schema_hash"]["api_version"] == mod.ADAPTER_API_VERSION
    assert artifact["baseline_import_and_output_receipts"]["import_ok"] is True
    assert artifact["default_off_compatibility_rows"]["all_public_outputs_match"] is True
    assert artifact["shadow_decision_rows"]["enabled_shadow_row_count"] > 0
    assert artifact["exact_write_admission_rows"]["all_writes_have_prior_exact_receipt"] is True
    assert artifact["persistence_rollback_and_tombstone_receipts"]["non_resurrection_after_load"] is True
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["attack_matrix"]["all_critical_attacks_fail_closed"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["preconditions_checked"]["planning_date"] == "20260821"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_learn_6479_attack_matrix_covers_required_boundaries(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6479-EXACT-ADMIT: all production-boundary attacks fail closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["readiness_promoted_attack_count"] == 0
    assert {row["status"] for row in attacks["rows"]} == {"failed_closed"}


def test_req_learn_6479_validation_recomputes_ready_and_fails_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6479: validation rejects hidden compatibility or authority failures."""

    artifact = _artifact(tmp_path)
    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data.__setitem__("field_principles", {})),
        ("field_provenance", lambda data: data.__setitem__("field_provenance", {})),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        (
            "default_off_compatibility",
            lambda data: data["default_off_compatibility_rows"].__setitem__(
                "all_public_outputs_match",
                False,
            ),
        ),
        (
            "exact_write_admission",
            lambda data: data["exact_write_admission_rows"].__setitem__(
                "all_writes_have_prior_exact_receipt",
                False,
            ),
        ),
        (
            "persistence",
            lambda data: data["persistence_rollback_and_tombstone_receipts"].__setitem__(
                "non_resurrection_after_load",
                False,
            ),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"].__setitem__("all_critical_attacks_fail_closed", False),
        ),
        (
            "aggregate",
            lambda data: data["aggregate_row_recomputation"].__setitem__("matches_reported", False),
        ),
        (
            "factor_cache_shadow_adapter_ready_score",
            lambda data: data.__setitem__("factor_cache_shadow_adapter_ready_score", 0.5),
        ),
        (
            "protected_files",
            lambda data: data["protected_files_unchanged"].__setitem__("unchanged", False),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "llm")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "ambiguous")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "checksum":
            if expected != "factor_cache_shadow_adapter_ready_score":
                bad["factor_cache_shadow_adapter_ready_score"] = mod.readiness_score(bad)
            if expected != "honest_verdict":
                bad["honest_verdict"] = mod.honest_verdict(bad)
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        errors = mod.artifact_errors(bad)
        assert any(expected in error for error in errors), (expected, errors)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6479_run_writes_stable_artifact_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6479: run and CLI write deterministic terminal JSON."""

    destination = tmp_path / "result.json"
    first = _artifact(tmp_path / "first", write=False)
    second = mod.run(
        date="20260821",
        root=REPO,
        result_path=destination,
        ledger_dir=tmp_path / "second" / "ledger",
        duration_s=1.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == second
    assert first == second

    called = {"run": 0}

    def fake_run(**_: object) -> dict[str, object]:
        called["run"] += 1
        return {}

    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main(["--date", "20260821"]) is None
    assert called["run"] == 1


def test_req_learn_6479_env_restore_branch_is_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6479: default-on env attack restores an existing value."""

    monkeypatch.setenv("CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER", "already-set")
    attacks = mod.attack_matrix(tmp_path / "attacks")
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert os.environ["CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER"] == "already-set"
