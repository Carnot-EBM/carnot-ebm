"""Tests for Exp5640 FR-11 shadow pipeline integration.

Spec refs: REQ-LEARN-5640,
SCENARIO-LEARN-5640-EQUIVALENCE,
SCENARIO-LEARN-5640-SHADOW,
SCENARIO-LEARN-5640-REPLAY,
SCENARIO-LEARN-5640-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5640_fr11_shadow_pipeline_integration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_fr11_shadow_adapter.py "
    "tests/python/test_experiment_5640_fr11_shadow_pipeline_integration.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/pipeline/fr11_shadow_adapter.py,"
    "python/carnot/experiment_5640_fr11_shadow_pipeline_integration.py "
    "-m pytest tests/python/test_fr11_shadow_adapter.py "
    "tests/python/test_experiment_5640_fr11_shadow_pipeline_integration.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/pipeline/fr11_shadow_adapter.py,"
    "python/carnot/experiment_5640_fr11_shadow_pipeline_integration.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5640_fr11_shadow_pipeline_integration.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]


def _artifact(tmp_path: Path) -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        ledger_dir=tmp_path / "ledgers",
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )


def test_req_learn_5640_spec_declares_shadow_pipeline_contract() -> None:
    """REQ-LEARN-5640: OpenSpec anchors gates, fields, scenarios, and substrate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5640") :]

    for marker in (
        "REQ-LEARN-5640",
        "SCENARIO-LEARN-5640-EQUIVALENCE",
        "SCENARIO-LEARN-5640-SHADOW",
        "SCENARIO-LEARN-5640-REPLAY",
        "SCENARIO-LEARN-5640-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "disabled by default",
        "exact verification remains authoritative",
        "append-only JSONL decision ledger",
        "checkpoint commit SHALL be atomic",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5640_artifact_fields_and_mechanical_ready_score(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5640-ARTIFACT: all readiness gates must pass."""

    artifact = _artifact(tmp_path)

    assert mod.validate_artifact(artifact) is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]

    assert artifact["openspec_requirement_ids"] == list(mod.SPEC_REFS)
    assert artifact["adapter_path"] == mod.ADAPTER_RELATIVE_PATH.as_posix()
    assert artifact["feature_flag"] == mod.FEATURE_FLAG
    assert artifact["default_enabled"] is False
    assert artifact["exact_verifier_authority"] is True
    assert artifact["shadow_decision_count"] > 0
    assert artifact["shadow_offline_parity"] is True
    assert artifact["default_path_equivalence"] is True
    assert artifact["unsafe_update_accept_count"] == 0
    assert artifact["checkpoint_atomicity_pass"] is True
    assert artifact["restart_replay_pass"] is True
    assert artifact["rollback_pass"] is True
    assert artifact["ledger_lineage_complete"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["fr11_shadow_integration_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["benefit_evidence_within_exp5639_bound"] is True
    assert artifact["upstream_gate_receipts"]["exp5639"]["promotion_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "shadow_ready_not_default_enabled" in artifact["honest_verdict"]


def test_scenario_learn_5640_replay_controls_are_exercised(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5640-REPLAY: adversarial stream controls fail closed."""

    artifact = _artifact(tmp_path)
    controls = artifact["replay_controls"]

    for control in (
        "crash_restart",
        "duplicate_delivery",
        "delayed_labels",
        "poison",
        "corrupted_checkpoint",
        "rollback",
        "inactive_adapter",
        "feature_disabled_equivalence",
    ):
        assert controls[control]["pass"] is True

    assert controls["duplicate_delivery"]["recommendation"] == "abstain"
    assert controls["delayed_labels"]["rollback_reason"] == "delayed_label_pending"
    assert controls["poison"]["rollback_reason"] == "poison_rejected"
    assert controls["corrupted_checkpoint"]["rollback_reason"] == "corrupted_checkpoint_recovered"
    assert controls["inactive_adapter"]["decision_count"] == 0
    assert controls["feature_disabled_equivalence"]["ledger_written"] is False
    assert artifact["shadow_offline_mismatches"] == []


def test_req_learn_5640_run_writes_stable_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-5640: run writes deterministic JSON with checksum."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    first = mod.run(
        root=REPO,
        result_path=destination,
        ledger_dir=tmp_path / "first-ledgers",
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    second = mod.run(
        root=REPO,
        result_path=tmp_path / "second.json",
        ledger_dir=tmp_path / "second-ledgers",
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )

    assert loaded == first
    assert first == second
    assert first["reproducibility_checksum"] == mod.reproducibility_checksum(first)
    assert mod.validate_artifact(first) is True


def test_req_learn_5640_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5640: schema and readiness failures are terminal blockers."""

    artifact = _artifact(tmp_path)
    assert any("missing required fields" in error for error in mod.artifact_errors({}))

    cases: list[tuple[str, dict[str, object]]] = []

    for principles in ({}, []):
        bad = deepcopy(artifact)
        bad["field_principles"] = principles
        cases.append(("field_principles", bad))

    for field, value, expected in (
        ("openspec_requirement_ids", [], "openspec_requirement_ids"),
        ("adapter_path", "wrong.py", "adapter_path"),
        ("feature_flag", "WRONG_FLAG", "feature_flag"),
        ("default_enabled", True, "default_enabled"),
        ("exact_verifier_authority", False, "exact_verifier_authority"),
        ("shadow_decision_count", 0, "shadow_decision_count"),
        ("shadow_offline_parity", False, "shadow_offline_parity"),
        ("default_path_equivalence", False, "default_path_equivalence"),
        ("unsafe_update_accept_count", 1, "unsafe_update_accept_count"),
        ("checkpoint_atomicity_pass", False, "checkpoint_atomicity_pass"),
        ("restart_replay_pass", False, "restart_replay_pass"),
        ("rollback_pass", False, "rollback_pass"),
        ("ledger_path", "", "ledger_path"),
        ("ledger_lineage_complete", False, "ledger_lineage_complete"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        (
            "inference_substrate",
            "llm_inference",
            "inference_substrate",
        ),
        ("random_seeds", [], "random_seeds"),
        ("benefit_evidence_within_exp5639_bound", False, "benefit_evidence"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["upstream_gate_receipts"]["exp5639"]["promotion_ready"] = False
    cases.append(("upstream_gate_receipts", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "ambiguous"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: stale"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    cases.append(("reproducibility_checksum", bad))

    bad = deepcopy(artifact)
    bad["fr11_shadow_integration_ready_score"] = 0.0
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    cases.append(("fr11_shadow_integration_ready_score", bad))

    bad = deepcopy(artifact)
    bad["replay_controls"]["duplicate_delivery"]["recommendation"] = "illegal"
    cases.append(("replay_controls", bad))

    for expected, bad_artifact in cases:
        errors = mod.artifact_errors(bad_artifact)
        assert any(expected in error for error in errors), (expected, errors)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)


def test_req_learn_5640_helper_edges_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5640: helper edge branches are deterministic."""

    target = tmp_path / "target.txt"
    tmp = tmp_path / "target.txt.tmp"
    target.write_text("old", encoding="utf-8")
    tmp.write_text("partial", encoding="utf-8")
    mod._reset_paths(target)
    assert not target.exists()
    assert not tmp.exists()

    root_ledger, root_checkpoint = mod._actual_paths(tmp_path, None)
    assert root_ledger == tmp_path / mod.LEDGER_RELATIVE_PATH
    assert root_checkpoint == tmp_path / mod.CHECKPOINT_RELATIVE_PATH

    assert mod.benefit_evidence_within_exp5639_bound({"exp5639": {"promotion_ready": False}}) is False

    parity, mismatches = mod.shadow_offline_parity([], [{"sequence": 0}])
    assert parity is False
    assert mismatches[0]["kind"] == "count"

    artifact_base = tmp_path / "mismatch"
    artifact = _artifact(artifact_base)
    rows = mod.load_ledger(artifact_base / "ledgers" / mod.LEDGER_RELATIVE_PATH.name)
    changed = [dict(row) for row in rows]
    changed[0]["recommendation"] = "abstain"
    parity, mismatches = mod.shadow_offline_parity(rows, changed)
    assert parity is False
    assert mismatches[0]["index"] == 0

    blocked = deepcopy(artifact)
    blocked["upstream_gate_receipts"]["exp5639"]["promotion_ready"] = False
    blocked["fr11_shadow_integration_ready_score"] = mod.readiness_score(blocked)
    assert mod.honest_verdict(blocked).startswith("blocked:")

    called = {"run": 0}

    def fake_run() -> dict[str, object]:
        called["run"] += 1
        return {}

    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main() is None
    assert called["run"] == 1
