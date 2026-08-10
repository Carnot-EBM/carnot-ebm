"""Tests for Exp6276 certified dual-cache admission.

Spec refs: REQ-LEARN-6276, SCENARIO-LEARN-6276-PARTITIONS,
SCENARIO-LEARN-6276-DUAL-CACHE, SCENARIO-LEARN-6276-CERTIFICATE,
SCENARIO-LEARN-6276-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6276_certified_dual_cache_admission as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["certified_admission_ready_score"] = mod.ready_score(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def _event(
    index: int,
    *,
    energy: float,
    unsafe: int = 0,
    family: str = "known_family",
    partition: str = "train",
    poisoned: bool = False,
) -> mod.EnergyEvent:
    return mod.EnergyEvent(
        row_id=f"row-{index}",
        event_id=f"event-{index}",
        model_hf_id="model-a",
        family=family,
        partition=partition,
        source_partition="calibration" if partition == "train" else "future_known",
        chronological_index=index,
        unsafe_label=unsafe,
        energy=energy,
        task_key=family,
        source_disposition="clean",
        content_addressed_row_id=f"sha256:{index:064x}",
        variant_kind="canonical",
        control_kind="normal",
        poisoned=poisoned,
    )


def test_req_6276_spec_declares_certified_dual_cache_contract() -> None:
    """REQ-LEARN-6276: OpenSpec owns the 6276 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6276") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6276-1",
        "REQ-LEARN-6276-7",
        "SCENARIO-LEARN-6276-PARTITIONS",
        "SCENARIO-LEARN-6276-DUAL-CACHE",
        "SCENARIO-LEARN-6276-CERTIFICATE",
        "SCENARIO-LEARN-6276-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6276_artifact_writes_required_receipts(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6276-PARTITIONS: gates freeze before held scoring."""

    artifact = _artifact(tmp_path, write=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["certified_admission_ready_score"] == 1.0
    assert artifact["source_mutation_count"] == 0
    assert type(artifact["source_mutation_count"]) is int
    assert artifact["weight_mutation_count"] == 0
    assert type(artifact["weight_mutation_count"]) is int
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    bridge = artifact["upstream_bridge_path_hash_and_terminal_class"]
    assert bridge["sha256"] == mod.EXPECTED_BRIDGE_SHA256
    assert bridge["exact_hash_matched"] is True
    assert bridge["terminal_class"] == "complete_ready"

    control = artifact["exp6264_control_path_hash_and_summary"]
    assert control["sha256"] == mod.EXPECTED_EXP6264_SHA256
    assert control["reproduced_global_threshold"]["threshold"] == pytest.approx(
        control["artifact_global_threshold"]["threshold"]
    )
    assert control["task_conditional_threshold_reused_as_treatment"] is False
    assert control["reproduced_global_threshold"]["held_fire_count"] == 180
    assert control["reproduced_global_threshold"]["shifted_unsafe_advice_count"] == 0

    partitions = artifact["frozen_train_validation_test_partitions"]
    assert partitions["row_count_by_partition"] == {
        "test": 160,
        "train": 192,
        "validation": 128,
    }
    assert partitions["reserve_row_count"] == 48
    assert partitions["held_labels_used_for_fit_count"] == 0

    reserve = artifact["frozen_reserve_manifest_path_and_hash"]
    reserve_path = Path(reserve["path"])
    assert reserve_path.exists()
    assert reserve["sha256"] == mod.sha256_file(reserve_path)

    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_6276_certified_dual_cache_is_safe_and_useful(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6276-DUAL-CACHE: treatment is gated and stratified."""

    artifact = _artifact(tmp_path)
    coverage = artifact["coverage_by_arm_partition_model_task_family"]
    unsafe = artifact["unsafe_advice_by_arm_partition_model_task_family"]
    calibration = artifact["calibration_and_abstention_by_arm"]
    utility = artifact["utility_and_negative_transfer_by_arm"]
    purity = artifact["cache_purity_and_redundancy_by_arm"]

    assert coverage["certified_dual_cache"]["held"]["fire_count"] == 80
    assert coverage["certified_dual_cache"]["validation"]["coverage"] == pytest.approx(0.625)
    assert coverage["certified_dual_cache"]["test"]["fire_count"] == 0
    assert unsafe["certified_dual_cache"]["test"]["unsafe_advice_count"] == 0
    assert unsafe["unconditional_cache"]["test"]["unsafe_advice_count"] == 60
    assert calibration["certified_dual_cache"]["held"]["abstention_rate"] == pytest.approx(
        208 / 288
    )
    assert utility["certified_dual_cache"]["test"]["negative_transfer_present"] is False

    assert purity["certified_dual_cache"]["positive_cache"]["record_count"] > 0
    assert purity["certified_dual_cache"]["positive_cache"]["unsafe_record_count"] == 0
    assert purity["certified_dual_cache"]["negative_cache"]["safe_record_count"] == 0
    assert purity["certified_dual_cache"]["redundant_candidate_count"] > 0

    intervals = artifact["paired_intervals_and_sample_sizes"]
    assert intervals["certified_dual_cache_vs_unconditional_shifted_unsafe_advice"]["n"] == 160
    assert intervals["certified_dual_cache_vs_global_held_utility"]["n"] == 288


def test_scenario_6276_gate_edges_fail_closed() -> None:
    """SCENARIO-LEARN-6276-DUAL-CACHE: edge rows cannot leak advice."""

    train = [
        _event(0, energy=-2.0, unsafe=0),
        _event(1, energy=-1.9, unsafe=0),
        _event(2, energy=2.0, unsafe=1),
        _event(3, energy=2.1, unsafe=1),
        _event(4, energy=-1.8, unsafe=0),
        _event(5, energy=-1.7, unsafe=0),
        _event(6, energy=2.2, unsafe=1),
        _event(7, energy=2.3, unsafe=1),
    ]
    fit = mod.fit_certified_dual_cache(train)
    safe = _event(10, energy=-1.85, unsafe=0, partition="validation")
    unsafe = _event(11, energy=2.2, unsafe=1, partition="validation")
    unseen = _event(12, energy=-1.85, unsafe=0, family="unseen", partition="validation")
    poisoned = _event(13, energy=-1.85, unsafe=0, partition="validation", poisoned=True)

    assert fit["reserve_certificate"]["certified"] is True
    assert mod.advice_fires("certified_dual_cache", safe, fit) is True
    assert mod.advice_fires("certified_dual_cache", unsafe, fit) is False
    assert mod.advice_fires("certified_dual_cache", unseen, fit) is False
    assert mod.advice_fires("certified_dual_cache", poisoned, fit) is False
    assert mod.advice_fires("exp6264_global_threshold", safe, fit) is True
    assert mod.advice_fires("no_cache", safe, fit) is False
    assert mod.advice_fires("unconditional_cache", poisoned, fit) is False
    assert mod._dual_cache_admits_without_certificate(poisoned, fit) is False
    assert mod._safe_probability(safe, None, 1.0) == 0.0

    tight_entropy = deepcopy(fit)
    tight_entropy["entropy_gate"]["entropy_threshold"] = 0.0
    assert mod._dual_cache_admits_without_certificate(safe, tight_entropy) is False

    skipped = mod._build_cache_records(
        [
            poisoned,
            _event(14, energy=0.1, unsafe=0),
        ],
        threshold=0.0,
        scale=1.0,
        entropy_threshold=1.0,
        diversity_gap=0.05,
    )
    assert skipped == ([], [], 0)

    assert mod._linear_fit([1.0], [0.0])["slope"] is None
    assert mod._linear_fit([1.0, 1.0], [0.0, 1.0])["slope"] is None

    empty_fit = mod.fit_certified_dual_cache([])
    assert empty_fit["reserve_certificate"]["certified"] is False
    assert mod.advice_fires("certified_dual_cache", safe, empty_fit) is False
    assert mod._admission_probability("exp6264_global_threshold", safe, empty_fit) == 0.0
    with pytest.raises(ValueError, match="unknown arm"):
        mod.advice_fires("mystery_arm", safe, fit)
    with pytest.raises(ValueError, match="unknown arm"):
        mod._admission_probability("mystery_arm", safe, fit)


def test_scenario_6276_certificate_controls_and_validation_guards(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6276-CERTIFICATE: unsafe certificates fail closed."""

    first = _artifact(tmp_path)
    second = _artifact(tmp_path)
    assert first == second
    assert mod.validate_artifact(first) is True

    timed = mod.run(result_path=tmp_path / "timed.json", test_exit_codes=_passing_exit_codes())
    assert timed["duration_s"] >= 0.001

    assert first["impurity_reproduction_number_upper_confidence_bound"] < 1.0
    assert first["poison_controls"]["passed"] is True
    assert first["drift_controls"]["passed"] is True
    assert first["rollback_identity_receipt"]["exact_rollback"] is True

    blocked = deepcopy(first)
    blocked["upstream_bridge_path_hash_and_terminal_class"]["exact_hash_matched"] = False
    _refresh(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    high_impurity = deepcopy(first)
    high_impurity["impurity_reproduction_number_upper_confidence_bound"] = 1.0
    _refresh(high_impurity)
    assert high_impurity["status"] == "complete_null"
    assert mod.validate_artifact(high_impurity) is True

    no_certificate = deepcopy(first)
    no_certificate["impurity_reproduction_number_upper_confidence_bound"] = None
    _refresh(no_certificate)
    assert "could not be estimated" in no_certificate["honest_verdict"]
    assert mod.validate_artifact(no_certificate) is True

    default_null = deepcopy(first)
    default_null["poison_controls"]["passed"] = False
    _refresh(default_null)
    assert default_null["honest_verdict"].endswith("readiness gate")
    assert mod.validate_artifact(default_null) is True

    failed_test = deepcopy(first)
    failed_test["test_exit_codes"][mod.GLOBAL_PYTEST_COMMAND] = 2
    _refresh(failed_test)
    assert failed_test["status"] == "complete_null"
    assert "recorded test command failed" in failed_test["honest_verdict"]
    assert mod.validate_artifact(failed_test) is True

    missing = dict(first)
    missing.pop("poison_controls")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(first)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_source = deepcopy(first)
    bad_source["source_mutation_count"] = {"value": 0}
    bad_source["reproducibility_checksum"] = mod.reproducibility_checksum(bad_source)
    with pytest.raises(ValueError, match="source_mutation_count"):
        mod.validate_artifact(bad_source)

    bad_weight = deepcopy(first)
    bad_weight["weight_mutation_count"] = 1
    bad_weight["reproducibility_checksum"] = mod.reproducibility_checksum(bad_weight)
    with pytest.raises(ValueError, match="weight_mutation_count"):
        mod.validate_artifact(bad_weight)

    bad_ready = deepcopy(first)
    bad_ready["coverage_by_arm_partition_model_task_family"]["certified_dual_cache"][
        "held"
    ]["fire_count"] = 0
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_ready)

    bad_status = deepcopy(first)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(first)
    bad_verdict["honest_verdict"] = "complete: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_principles = deepcopy(first)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_provenance_type = deepcopy(first)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance_field = deepcopy(first)
    bad_provenance_field["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance_field["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_field
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_field)

    bad_arm = deepcopy(first)
    bad_arm["arm_definitions"]["arm_names"] = []
    bad_arm["reproducibility_checksum"] = mod.reproducibility_checksum(bad_arm)
    with pytest.raises(ValueError, match="arm definition"):
        mod.validate_artifact(bad_arm)
