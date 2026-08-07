"""Tests for Exp6180 Exp6166 reproducibility adjudication.

Spec refs: REQ-SAMPLE-6180, REQ-SAMPLE-6180-IMMUTABLE-SOURCE,
REQ-SAMPLE-6180-NO-REFIT, REQ-SAMPLE-6180-STOCHASTIC-REPLAY,
REQ-SAMPLE-6180-ISOLATED-TESTS, REQ-SAMPLE-6180-OLD-FAILURE-CLASSIFICATION,
REQ-SAMPLE-6180-NO-HARDWARE-PROMOTION, REQ-SAMPLE-6180-PROTECTED-FILE,
REQ-SAMPLE-6180-CHECKSUM, SCENARIO-SAMPLE-6180-READ-ONLY-ADJUDICATION,
SCENARIO-SAMPLE-6180-HISTORICAL-BLOCK-PRESERVED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6166_mode_jumping_factor_thermalization as exp6166
from carnot import experiment_6180_exp6166_reproducibility_adjudication as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _command_receipts() -> list[dict[str, object]]:
    return [
        {
            "name": "focused_exp6180_tests",
            "command": (
                "JAX_PLATFORMS=cpu .venv/bin/pytest "
                "tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py "
                "-q --no-cov -n 0"
            ),
            "exit_code": 0,
            "stdout": "4 passed",
            "stderr": "",
        },
        {
            "name": "new_code_coverage",
            "command": (
                "JAX_PLATFORMS=cpu .venv/bin/coverage run --source=python/carnot "
                "-m pytest tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py "
                "-q --no-cov -n 0 && .venv/bin/coverage report "
                "--include=python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py "
                "--fail-under=100"
            ),
            "exit_code": 0,
            "stdout": "100% coverage",
            "stderr": "",
        },
        {
            "name": "required_full_python_suite_once",
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 2,
            "stdout": "same unrelated pre-existing suite state",
            "stderr": "tracked result mutations and unrelated collection errors",
            "classification": "unrelated_preexisting",
        },
    ]


def test_req_sample_6180_spec_declares_immutable_read_only_contract() -> None:
    """REQ-SAMPLE-6180: OpenSpec names every Exp6180 preservation gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6180") :]
    section = section[: section.index("## Implementation Status (REQ-SAMPLE-6180)")]

    for marker in (
        "REQ-SAMPLE-6180-IMMUTABLE-SOURCE",
        "REQ-SAMPLE-6180-NO-REFIT",
        "REQ-SAMPLE-6180-STOCHASTIC-REPLAY",
        "REQ-SAMPLE-6180-ISOLATED-TESTS",
        "REQ-SAMPLE-6180-OLD-FAILURE-CLASSIFICATION",
        "REQ-SAMPLE-6180-NO-HARDWARE-PROMOTION",
        "REQ-SAMPLE-6180-PROTECTED-FILE",
        "REQ-SAMPLE-6180-CHECKSUM",
        "SCENARIO-SAMPLE-6180-READ-ONLY-ADJUDICATION",
        "SCENARIO-SAMPLE-6180-HISTORICAL-BLOCK-PRESERVED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_6180_recomputes_from_artifact_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-6180-READ-ONLY-ADJUDICATION: no training call is needed."""

    def fail_if_refit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("REQ-SAMPLE-6180-NO-REFIT forbids Exp6166 training")

    monkeypatch.setattr(exp6166, "_train_probabilities", fail_if_refit)
    artifact = mod.build_adjudication_artifact(
        command_receipts=_command_receipts(),
        duration_s=0.0,
        before_snapshot=mod.snapshot_preconditions(REPO),
    )

    assert artifact["status"] == "complete_positive"
    assert artifact["no_refit_receipt"]["training_functions_invoked"] is False
    assert artifact["no_refit_receipt"]["metric_source"] == "historical_artifact_probabilities"
    assert artifact["stochastic_replay_receipt"]["seeds"] == list(exp6166.TRAINING_SEEDS)
    assert artifact["stochastic_replay_receipt"]["all_pair_hashes_match"] is True
    assert artifact["recomputed_metric_determination"]["all_declared_metrics_match"] is True
    assert artifact["recomputed_metric_determination"]["mode_jump_improved_over_local_only"] is True
    assert artifact["companion_determination"]["historical_exp6166_status_preserved"] == "blocked"
    assert artifact["companion_determination"]["adjudicated_result"] == (
        "software_only_positive_reproducible"
    )
    assert artifact["no_hardware_promotion_receipt"]["hardware_execution_claimed"] is False


def test_scenario_6180_writes_schema_checksum_and_preserves_old_block(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-6180-HISTORICAL-BLOCK-PRESERVED: output is auditable."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    before = mod.snapshot_preconditions(REPO)
    artifact = mod.write_adjudication_artifact(
        output_path=output,
        command_receipts=_command_receipts(),
        duration_s=0.0,
        before_snapshot=before,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["old_full_suite_failure_classification"]["historical_exit_code"] == 2
    assert artifact["old_full_suite_failure_classification"]["classification"] == (
        "unrelated_preexisting_repository_suite_failure"
    )
    assert artifact["old_full_suite_failure_classification"]["exp6166_status_after"] == "blocked"
    assert artifact["before_after_byte_comparison"]["exp6166_artifact"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_req_sample_6180_defensive_schema_and_classification_edges() -> None:
    """REQ-SAMPLE-6180-CHECKSUM: bad receipts fail closed."""

    artifact = mod.build_adjudication_artifact(
        command_receipts=_command_receipts(),
        duration_s=0.0,
        before_snapshot=mod.snapshot_preconditions(REPO),
    )

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    hardware = deepcopy(artifact)
    hardware["no_hardware_promotion_receipt"]["hardware_execution_claimed"] = True
    hardware["honest_verdict"] = mod.honest_verdict(hardware)
    hardware["reproducibility_checksum"] = mod.reproducibility_checksum(hardware)
    with pytest.raises(ValueError, match="hardware"):
        mod.validate_artifact(hardware)

    performance = deepcopy(artifact)
    performance["no_hardware_promotion_receipt"]["latency_power_energy_and_speedup_claimed"] = True
    performance["honest_verdict"] = mod.honest_verdict(performance)
    performance["reproducibility_checksum"] = mod.reproducibility_checksum(performance)
    with pytest.raises(ValueError, match="hardware_performance"):
        mod.validate_artifact(performance)

    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "gpu"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(substrate)

    status = deepcopy(artifact)
    status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status)

    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "complete_positive: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum)

    provenance_type = deepcopy(artifact)
    provenance_type["field_provenance"] = []
    provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(provenance_type)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_type)

    provenance = deepcopy(artifact)
    provenance["field_provenance"]["status"]["principle"] = "wrong"
    provenance["reproducibility_checksum"] = mod.reproducibility_checksum(provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance)

    no_old_failure = {
        "status": "blocked",
        "test_exit_codes": {command: 0 for command in exp6166.DEFAULT_TEST_COMMANDS},
    }
    classified = mod.classify_old_full_suite_failure(
        no_old_failure,
        {"canary_failure_classification": {"rows": []}},
    )
    assert classified["classification"] == "no_historical_full_suite_failure"

    unclassified = mod.classify_old_full_suite_failure(
        {"status": "blocked", "test_exit_codes": {exp6166.GLOBAL_PYTEST_COMMAND: 1}},
        {"canary_failure_classification": {"rows": []}},
    )
    assert unclassified["classification"] == "unclassified_historical_full_suite_failure"
    assert mod._metric_delta(float("inf"), float("inf")) == 0.0
