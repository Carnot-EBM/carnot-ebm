"""Tests for the recovered V622 entrance-bank support audit.

Spec refs: REQ-VERIFY-7093 and SCENARIO-VERIFY-7093-*.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from carnot import experiment_7087_v621_entrance_bank_sufficiency_audit as legacy
from carnot import experiment_7093_v622_entrance_bank_sufficiency_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TEST_PATH = (
    REPO_ROOT / "tests/python/test_experiment_7087_v621_entrance_bank_sufficiency_audit.py"
)


def _legacy_test_helpers() -> ModuleType:
    """Load the existing test data builders so Exp7093 reuses their tested science."""

    spec = importlib.util.spec_from_file_location("exp7087_test_helpers_for_7093", LEGACY_TEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _focused_test_row() -> dict:
    """Build one successful receipt with the exact command required by the validator."""

    return {
        "command": audit.focused_test_command(),
        "coverage_report_command": audit.focused_coverage_report_command(new_code_only=False),
        "new_code_coverage_report_command": audit.focused_coverage_report_command(
            new_code_only=True
        ),
        "test_files": list(audit.FOCUSED_TEST_FILES),
        "coverage_scopes": list(audit.FOCUSED_COVERAGE_SCOPES),
        "coverage_fail_under": 100,
        "returncode": 0,
        "test_returncode": 0,
        "coverage_returncode": 0,
        "new_code_coverage_returncode": 0,
        "stdout_hash": "sha256:" + "1" * 64,
        "stderr_hash": "sha256:" + "2" * 64,
        "passed": True,
    }


def _valid_artifact() -> dict:
    """Build a small valid promotion while keeping the full Exp7087 row logic."""

    helpers = _legacy_test_helpers()
    helpers.MODELS = audit.REQUIRED_SOURCE_MODEL_IDS
    fixture = helpers._fixture()
    bank = helpers._bank(fixture)
    schema = audit.build_required_support_schema(
        fixture,
        audit.REQUIRED_SOURCE_MODEL_IDS,
        helpers.SEEDS,
    )
    replay = audit.recompute_audit(bank, fixture, schema)
    attacks = audit.run_counterfactual_attacks(bank, fixture, schema)
    return audit.build_artifact(
        run_date=audit.RUN_DATE,
        duration_s=1.25,
        bank=bank,
        fixture=fixture,
        preconditions={"all_passed": True, "checks": []},
        source_artifact_hashes={
            audit.BANK_PATH.name: audit.PINNED_BANK_SHA256,
            audit.FIXTURE_PATH.name: audit.PINNED_FIXTURE_SHA256,
        },
        replay=replay,
        fresh_process_rows=[{"passed": True, "fresh_process": True}],
        focused_test_rows=[_focused_test_row()],
        minimum_headroom_units=1,
        counterfactual_swap_rows=attacks,
    )


def test_identity_pins_and_required_fields_are_promoted() -> None:
    """REQ-VERIFY-7093 binds the new identity to the exact reviewed inputs."""

    assert audit.EXPERIMENT_ID == "experiment_7093_v622_entrance_bank_sufficiency_audit"
    assert audit.SCHEMA == "carnot.experiment_7093.v622_entrance_bank_sufficiency_audit.v1"
    assert audit.RUN_DATE == "20260907"
    assert audit.PINNED_BANK_SHA256 == (
        "sha256:4f9e73adfc2ce0ede707424f6f644705557faa27d4485fb2293f652ab1713495"
    )
    assert audit.PINNED_FIXTURE_SHA256 == (
        "sha256:6b62768e3387d40eebf462c199aab6a440321aa4a1549ff7d54faaba312f2277"
    )
    assert set(audit.REQUIRED_ARTIFACT_FIELDS) == set(audit.FIELD_PRINCIPLES)
    assert {"execution_venue", "source_model_specs", "focused_test_rows"} <= set(
        audit.REQUIRED_ARTIFACT_FIELDS
    )


def test_focused_coverage_command_scopes_both_audit_modules() -> None:
    """SCENARIO-VERIFY-7093-COVERAGE avoids the repository-wide denominator."""

    command = audit.focused_test_command()
    assert command[:5] == [audit.sys.executable, "-m", "coverage", "run", command[4]]
    assert command[4] == f"--include={','.join(audit.FOCUSED_COVERAGE_FILES)}"
    assert command[5:7] == ["-m", "pytest"]
    assert set(audit.FOCUSED_TEST_FILES) <= set(command)
    report = audit.focused_coverage_report_command(new_code_only=False)
    new_report = audit.focused_coverage_report_command(new_code_only=True)
    assert report[-1] == "--show-missing"
    assert new_report[-1] == "--fail-under=100"
    assert report != new_report


def test_promoted_artifact_has_new_identity_models_and_terminal_class() -> None:
    """SCENARIO-VERIFY-7093-PROMOTION keeps detailed replay under the new identity."""

    artifact = _valid_artifact()
    assert artifact["schema"] == audit.SCHEMA
    assert artifact["experiment_id"] == audit.EXPERIMENT_ID
    assert artifact["execution_venue"] == "host"
    assert [row["hf_id"] for row in artifact["source_model_specs"]] == list(
        audit.REQUIRED_SOURCE_MODEL_IDS
    )
    assert artifact["fresh_process_rows"]
    assert artifact["parse_recomputation_rows"]
    assert artifact["label_recomputation_rows"]
    assert artifact["causal_witness_replay_rows"]
    assert audit.validate_artifact(artifact) == []


def test_complete_insufficiency_is_terminal_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7093-TERMINAL maps completed low headroom to null, not partial."""

    base = {
        "entrance_support_audit_ready_score": 1,
        "entrance_selector_headroom_ready_score": 0,
        "verdict_class": "partial",
        "honest_verdict": "partial: support ready but selector headroom is insufficient",
        "reproducibility_checksum": "",
    }
    monkeypatch.setattr(legacy, "build_artifact", lambda **_kwargs: deepcopy(base))
    artifact = audit.build_artifact(
        run_date=audit.RUN_DATE,
        duration_s=0.1,
        bank={"model_specs": []},
        fixture={},
        preconditions={"all_passed": True, "checks": []},
        source_artifact_hashes={},
        replay={},
        fresh_process_rows=[],
        focused_test_rows=[],
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("null:")


@pytest.mark.parametrize(
    "check",
    [
        "entrance_proposal_bank_source_hash",
        "entrance_proposal_bank_complete_score",
        "entrance_fixture_source_hash",
        "entrance_fixture_ready_score",
        "checkpoint_readability",
        "raw_checkpoint_readability",
        "source_model_identities",
        "legacy_small_models_absent",
        "result_path_absent",
        "isolated_audit_paths_writable",
    ],
)
def test_each_precondition_path_builds_an_exact_terminal_block(check: str) -> None:
    """SCENARIO-VERIFY-7093-PREFLIGHT records every blocked path exactly."""

    checks = [audit.gate_row(check, True, False, False)]
    artifact = audit.build_blocked_artifact(
        run_date=audit.RUN_DATE,
        duration_s=0.1,
        preconditions={"all_passed": False, "checks": checks},
        source_artifact_hashes={},
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == check
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert artifact["focused_test_rows"] == []
    assert audit.validate_artifact(artifact) == []


def test_source_model_precondition_rejects_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7093-PREFLIGHT blocks missing and legacy-small model identities."""

    identities = [
        {"model_id": model_id, "passed": True, "identity_matches": True}
        for model_id in audit.REQUIRED_SOURCE_MODEL_IDS
    ]
    bank = {
        "model_specs": [{"hf_id": model_id} for model_id in audit.REQUIRED_SOURCE_MODEL_IDS],
        "model_identity_rows": identities,
    }
    monkeypatch.setattr(
        legacy,
        "collect_preconditions",
        lambda **_kwargs: {
            "all_passed": True,
            "checks": [],
            "bank": deepcopy(bank),
            "checkpoint_receipts": [{"passed": True}] * 6,
        },
    )
    passed = audit.collect_preconditions(
        bank_path=tmp_path / "bank.json",
        fixture_path=tmp_path / "fixture.json",
        result_path=tmp_path / "result.json",
        audit_root=tmp_path / "audit",
    )
    assert passed["all_passed"] is True
    assert passed["checks"][-3]["check"] == "source_model_identities"
    changed = deepcopy(bank)
    changed["model_specs"][-1]["hf_id"] = "Qwen/Qwen3.5-0.8B"
    monkeypatch.setattr(
        legacy,
        "collect_preconditions",
        lambda **_kwargs: {
            "all_passed": True,
            "checks": [],
            "bank": changed,
            "checkpoint_receipts": [{"passed": True}] * 6,
        },
    )
    blocked = audit.collect_preconditions(
        bank_path=tmp_path / "bank.json",
        fixture_path=tmp_path / "fixture.json",
        result_path=tmp_path / "result.json",
        audit_root=tmp_path / "audit",
    )
    assert blocked["all_passed"] is False
    assert (
        next(row for row in blocked["checks"] if row["check"] == "source_model_identities")[
            "passed"
        ]
        is False
    )
    assert (
        next(row for row in blocked["checks"] if row["check"] == "legacy_small_models_absent")[
            "passed"
        ]
        is False
    )


def test_six_counterfactual_attacks_include_duplicate_and_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7093-ATTACKS keeps all required mutations explicit."""

    base_rows = [
        {
            "attack": name,
            "baseline_raw_row_count": 2,
            "attacked_raw_row_count": 1 if name == "family_deletion" else 2,
            "pooled_row_count_preserved": name != "family_deletion",
            "attack_detected": True,
            "failed_checks": [name],
        }
        for name in ("family_deletion", "source_swap", "raw_byte_mutation", "label_conflict")
    ]
    observed_banks = []

    def recompute(changed: dict, _fixture: dict, _schema: dict) -> dict:
        observed_banks.append(changed)
        return {
            "errors": ["targeted_mutation"],
            "authenticity_passed": len(observed_banks) == 1,
            "family_sufficiency_passed": False,
            "conflict_resolution_passed": True,
            "leakage_passed": len(observed_banks) == 1,
        }

    monkeypatch.setattr(legacy, "run_counterfactual_attacks", lambda *_args: deepcopy(base_rows))
    monkeypatch.setattr(legacy, "recompute_audit", recompute)
    bank = {
        "raw_proposal_rows": [
            {"raw_key": "a", "generation_config": {}},
            {"raw_key": "b", "generation_config": {}},
        ]
    }
    rows = audit.run_counterfactual_attacks(bank, {}, {})
    assert {row["attack"] for row in rows} == {
        "family_deletion",
        "source_swap",
        "raw_byte_mutation",
        "duplicate",
        "conflict",
        "future_label_leakage",
    }
    assert all(row["attack_detected"] for row in rows)
    assert (
        next(row for row in rows if row["attack"] == "duplicate")["pooled_row_count_preserved"]
        is True
    )
    assert (
        next(row for row in rows if row["attack"] == "future_label_leakage")[
            "pooled_row_count_preserved"
        ]
        is True
    )
    assert [row["raw_key"] for row in observed_banks[0]["raw_proposal_rows"]] == ["a", "a"]
    assert observed_banks[1]["raw_proposal_rows"][0]["generation_config"]["future_label"] == {
        "reachable": True
    }


def test_fresh_replay_routes_through_exp7093(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7093-PROMOTION uses a distinct Exp7093 worker process."""

    helpers = _legacy_test_helpers()
    fixture = helpers._fixture()
    bank = helpers._bank(fixture)
    bank_path = tmp_path / "bank.json"
    fixture_path = tmp_path / "fixture.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    replay, rows = audit.run_fresh_replay(bank_path, fixture_path, tmp_path / "audit")
    assert replay["errors"] == []
    assert rows[0]["fresh_process"] is True
    assert rows[0]["passed"] is True
    assert "carnot.experiment_7093_v622_entrance_bank_sufficiency_audit" in rows[0]["command"]


def test_fresh_replay_reports_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7093-PROMOTION does not hide a failed fresh worker."""

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=9, stderr="worker failed"),
    )
    with pytest.raises(RuntimeError, match="fresh_replay_failed:returncode=9"):
        audit.run_fresh_replay(
            tmp_path / "bank.json",
            tmp_path / "fixture.json",
            tmp_path / "audit",
        )


def test_validator_requires_an_artifact_object() -> None:
    """REQ-VERIFY-7093 rejects a non-object result before field reduction."""

    assert audit.validate_artifact([]) == ["artifact_object_required"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda value: value.__setitem__("experiment_id", legacy.EXPERIMENT_ID),
            "identity_mismatch",
        ),
        (lambda value: value.__setitem__("execution_venue", "gpu"), "execution_venue_mismatch"),
        (
            lambda value: value["source_artifact_hashes"].__setitem__(
                audit.BANK_PATH.name, "sha256:" + "0" * 64
            ),
            "upstream_hash_mismatch",
        ),
        (lambda value: value.__setitem__("verdict_class", "partial"), "verdict_class_mismatch"),
        (lambda value: value.__setitem__("focused_test_rows", []), "focused_test_rows_mismatch"),
        (
            lambda value: value.__setitem__("counterfactual_swap_rows", []),
            "counterfactual_attack_rows_mismatch",
        ),
    ],
)
def test_validator_rejects_promoted_contract_mutations(mutation, expected_error: str) -> None:
    """REQ-VERIFY-7093 cold validation rejects changed promotion evidence."""

    artifact = _valid_artifact()
    mutation(artifact)
    artifact["reproducibility_checksum"] = audit.artifact_checksum(artifact)
    assert expected_error in audit.validate_artifact(artifact)
