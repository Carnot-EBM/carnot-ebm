"""Tests for Exp6237 activated mode-jump sampler A/B.

Spec refs: REQ-SAMPLER-6237,
SCENARIO-SAMPLER-6237-ACTIVATED-EQUIVALENCE,
SCENARIO-SAMPLER-6237-CONTROLS-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6237_activated_mode_jump_sampler_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    commands = (
        ".venv/bin/pytest tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py -q -o addopts=",
        ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py -m pytest tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py -q --no-cov -o addopts=",
        ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py --fail-under=100",
        ".venv/bin/python -m carnot.experiment_6237_activated_mode_jump_sampler_ab --date 20260809",
        ".venv/bin/pytest tests/python/test_e2e_serialization.py tests/python/test_pyo3_integration.py -q -o addopts=",
        ".venv/bin/python scripts/adversarial_verify.py results/experiment_6237_activated_mode_jump_sampler_ab.json",
        ".venv/bin/pytest tests/python -q",
    )
    return [
        {
            "name": f"cmd_{index}",
            "command": command,
            "exit_code": 0,
            "task_owned": True,
            "classification": "task_owned"
            if "tests/python -q" not in command
            else "repository_validation",
        }
        for index, command in enumerate(commands)
    ]


def test_req_sampler_6237_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-SAMPLER-6237: OpenSpec anchors every artifact field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6237") :]
    normalized = " ".join(section.split())

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "SCENARIO-SAMPLER-6237-ACTIVATED-EQUIVALENCE",
        "SCENARIO-SAMPLER-6237-CONTROLS-FAIL-CLOSED",
        "instrument_failure",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section


def test_req_sampler_6237_artifact_schema_and_no_claim_gates(tmp_path: Path) -> None:
    """REQ-SAMPLER-6237: the terminal JSON validates and stays software-only."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_equivalence_supported"
    assert artifact["honest_verdict"].startswith("complete_equivalence_supported:")
    assert artifact["equivalence_bounds_and_decision"]["decision"] == "equivalence_supported"
    assert artifact["default_off_preserved"] is True
    assert type(artifact["hardware_claim_count"]) is int
    assert artifact["hardware_claim_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["value"] is True


def test_scenario_sampler_6237_activation_gates_quality() -> None:
    """SCENARIO-SAMPLER-6237-ACTIVATED-EQUIVALENCE: activation is measured first."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )

    activation = artifact["treatment_activation_score"]
    jump_counts = artifact["jump_proposal_acceptance_and_transition_counts"]
    positive = artifact["multimodal_positive_control"]

    assert activation["activation_passed"] is True
    assert activation["score"] == pytest.approx(1.0)
    assert activation["mode_jump_proposal_count"] > 0
    assert activation["mode_jump_acceptance_count"] > 0
    assert positive["passed"] is True
    assert positive["quality_conclusions_allowed"] is True
    assert jump_counts["main_cells_all_recorded"] is True
    for chain in jump_counts["chains"]:
        assert chain["sample_labels"]
        assert chain["transition_counts"]
        if chain["arm"] == "mode_jump_runtime":
            assert chain["active_backend"] == "rust_pyo3"


def test_scenario_sampler_6237_support_parity_default_off_and_costs() -> None:
    """REQ-SAMPLER-6237-MATCHED-AB: support, parity, and costs are explicit."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )

    assert artifact["arm_support_matrix"]["all_main_fixture_arms_supported"] is True
    assert artifact["arm_support_matrix"]["unsupported_controls_fail_closed"] is True
    assert artifact["fallback_parity_control"]["exact_replay_match"] is True
    assert artifact["fallback_parity_control"]["rust_python_decision_logs_match"] is True
    assert artifact["wall_and_transition_costs"]["matched_transition_budget"] is True
    assert artifact["wall_and_transition_costs"]["all_main_cells_within_wall_budget"] is True
    assert artifact["preconditions_checked"]["computed_before_sampler_chains"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True


def test_scenario_sampler_6237_controls_record_failures_without_main_claims() -> None:
    """SCENARIO-SAMPLER-6237-CONTROLS-FAIL-CLOSED: controls are visible."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    controls = {row["cell"]: row for row in artifact["unsupported_or_failed_cells"]}

    assert artifact["arm_support_matrix"]["main_unsupported_or_failed_cells"] == []
    assert controls["unsupported_fixture_control"]["recorded"] is True
    assert controls["unsupported_fixture_control"]["classification"] == "unsupported_control"
    assert controls["zero_activation_control"]["decision"] == "instrument_failure"
    assert controls["degenerate_chain_control"]["recorded"] is True
    assert controls["interruption_restart_control"]["restart_equivalence"] is True
    assert controls["interruption_restart_control"]["interruption_recorded"] is True


def test_req_sampler_6237_inactive_treatment_is_instrument_failure() -> None:
    """REQ-SAMPLER-6237-ACTIVATION: inactive treatment is not a null verdict."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    inactive = deepcopy(artifact)
    inactive["treatment_activation_score"]["score"] = 0.0
    inactive["treatment_activation_score"]["activation_passed"] = False
    inactive["treatment_activation_score"]["mode_jump_proposal_count"] = 0
    inactive["treatment_activation_score"]["mode_jump_acceptance_count"] = 0
    inactive["multimodal_positive_control"]["passed"] = False
    inactive["multimodal_positive_control"]["quality_conclusions_allowed"] = False
    inactive["equivalence_bounds_and_decision"] = mod.equivalence_bounds_and_decision(inactive)
    inactive["sampler_runtime_ready_score"] = mod.sampler_runtime_ready_score(inactive)
    inactive["status"] = mod.status(inactive)
    inactive["honest_verdict"] = mod.honest_verdict(inactive)
    inactive["reproducibility_checksum"] = mod.reproducibility_checksum(inactive)

    assert inactive["status"] == "instrument_failure"
    assert inactive["equivalence_bounds_and_decision"]["decision"] == "instrument_failure"
    assert "null" not in inactive["honest_verdict"]
    assert mod.validate_artifact(inactive) is True


def test_req_sampler_6237_validate_artifact_rejects_bad_fields() -> None:
    """REQ-SAMPLER-6237: schema mutations fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    mutations = [
        ("hardware_claim_count", lambda data: data.__setitem__("hardware_claim_count", True)),
        ("default_off_preserved", lambda data: data.__setitem__("default_off_preserved", False)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "gpu")),
        (
            "equivalence_bounds_and_decision",
            lambda data: data["equivalence_bounds_and_decision"].__setitem__(
                "decision", "inconclusive"
            ),
        ),
        (
            "sampler_runtime_ready_score",
            lambda data: data.__setitem__("sampler_runtime_ready_score", 0.0),
        ),
        ("status", lambda data: data.__setitem__("status", "complete_null")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "bad")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("status", "wrong")),
        (
            "field_provenance",
            lambda data: data["field_provenance"]["status"].__setitem__("source", ""),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = []
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    inactive_bad = deepcopy(artifact)
    inactive_bad["treatment_activation_score"]["activation_passed"] = False
    inactive_bad["honest_verdict"] = "complete_null: bad"
    inactive_bad["reproducibility_checksum"] = mod.reproducibility_checksum(inactive_bad)
    with pytest.raises(ValueError, match="inactive_treatment_instrument_failure"):
        mod.validate_artifact(inactive_bad)


def test_req_sampler_6237_helper_edges_for_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLER-6237-CONTROLS: helper edge cases stay deterministic."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod._interval([0.25]) == [0.25, 0.25]  # noqa: SLF001
    assert mod._interval([]) == [0.0, 0.0]  # noqa: SLF001
    assert mod._mean([]) == 0.0  # noqa: SLF001
    assert mod._quality_from_labels(["left_peak"])["effective_sample_size"] == 1.0  # noqa: SLF001
    assert mod._quality_from_labels([])["degenerate"] is True  # noqa: SLF001
    assert mod._mode_coverage_fraction([]) == 0.0  # noqa: SLF001
    assert mod._support_error(RuntimeError("boom"))["error_type"] == "RuntimeError"  # noqa: SLF001
    assert "checkpoint" in mod._descriptor(6237, checkpoint={"state": {}})  # noqa: SLF001

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(bad_json)  # noqa: SLF001

    labels, target, _proposal = mod.frozen_mode_jump_inputs(REPO)
    missing_sample = mod._distribution_metrics(labels, target, ["left_peak"])  # noqa: SLF001
    assert missing_sample["kl_target_to_empirical"] == float("inf")

    assert (
        mod._all_quality_intervals_within_bounds(  # noqa: SLF001
            {"total_variation_to_target_delta": {"mean_95_interval": [2.0, 2.0]}}
        )
        is False
    )

    receipts_path = tmp_path / "receipts.json"
    receipts_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6237_COMMAND_RECEIPTS", str(receipts_path))
    assert mod._external_command_receipts() == _passing_receipts()  # noqa: SLF001

    missing_default = tmp_path / "missing.json"
    monkeypatch.delenv("CARNOT_6237_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing_default)
    assert mod._external_command_receipts() == []  # noqa: SLF001

    bad_receipts = tmp_path / "bad_receipts.json"
    bad_receipts.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6237_COMMAND_RECEIPTS", str(bad_receipts))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_command_receipts()  # noqa: SLF001

    class FakeBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run_descriptor(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

    real_backend = mod.ModeJumpRustBackend
    monkeypatch.setattr(mod, "ModeJumpRustBackend", FakeBackend)
    assert mod._unsupported_fixture_control(REPO)["fail_closed"] is False  # noqa: SLF001
    monkeypatch.setattr(mod, "ModeJumpRustBackend", real_backend)

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    blocked = deepcopy(artifact)
    blocked["task_owned_and_preexisting_nonzero_command_ledger"]["task_owned_failure_count"] = 1
    assert mod.status(blocked) == "blocked"

    unsupported = deepcopy(artifact)
    unsupported["arm_support_matrix"]["all_main_fixture_arms_supported"] = False
    unsupported["equivalence_bounds_and_decision"] = mod.equivalence_bounds_and_decision(
        unsupported
    )
    assert unsupported["equivalence_bounds_and_decision"]["decision"] == "inconclusive"

    quality_bad = deepcopy(artifact)
    quality_bad["distribution_quality_by_fixture_arm"]["all_supported_quality_passed"] = False
    quality_bad["equivalence_bounds_and_decision"] = mod.equivalence_bounds_and_decision(
        quality_bad
    )
    assert quality_bad["equivalence_bounds_and_decision"]["decision"] == "negative"

    interval_bad = deepcopy(artifact)
    interval_bad["paired_intervals"]["intervals"]["total_variation_to_target_delta"][
        "mean_95_interval"
    ] = [0.5, 0.5]
    interval_bad["equivalence_bounds_and_decision"] = mod.equivalence_bounds_and_decision(
        interval_bad
    )
    assert interval_bad["equivalence_bounds_and_decision"]["decision"] == "inconclusive"

    positive = deepcopy(artifact)
    positive["equivalence_bounds_and_decision"]["decision"] = "positive"
    assert mod.status(positive) == "complete_positive"
    negative = deepcopy(artifact)
    negative["equivalence_bounds_and_decision"]["decision"] = "negative"
    assert mod.status(negative) == "complete_negative"
    inconclusive = deepcopy(artifact)
    inconclusive["equivalence_bounds_and_decision"]["decision"] = "inconclusive"
    assert mod.status(inconclusive) == "complete_inconclusive"

    monkeypatch.delenv("CARNOT_6237_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda **_kwargs: {
            "status": "complete_equivalence_supported",
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
    )
    assert mod.main(["--date", "20260809"]) == 0
    assert "complete_equivalence_supported" in capsys.readouterr().out
