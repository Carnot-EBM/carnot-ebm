"""Tests for Exp5639 anytime-valid CSL independent audit.

Spec refs: REQ-LEARN-5639,
SCENARIO-LEARN-5639-GATES,
SCENARIO-LEARN-5639-RECOMPUTE,
SCENARIO-LEARN-5639-ANYTIME,
SCENARIO-LEARN-5639-ADVERSARIAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5639_anytime_valid_csl_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5639_anytime_valid_csl_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5639_anytime_valid_csl_independent_audit.py "
    "-m pytest tests/python/test_experiment_5639_anytime_valid_csl_independent_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5639_anytime_valid_csl_independent_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5639_anytime_valid_csl_independent_audit.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-LEARN-5639: build the replay once from exact local receipts."""

    return mod.build_artifact(
        root=REPO,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path_factory.mktemp("exp5639_checkpoints"),
    )


def test_req_learn_5639_spec_declares_anytime_independent_audit_contract() -> None:
    """REQ-LEARN-5639: OpenSpec anchors gates, fields, substrate, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5639") :]

    for marker in (
        "REQ-LEARN-5639",
        "SCENARIO-LEARN-5639-GATES",
        "SCENARIO-LEARN-5639-RECOMPUTE",
        "SCENARIO-LEARN-5639-ANYTIME",
        "SCENARIO-LEARN-5639-ADVERSARIAL",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "Exp 5628",
        "Exp 5638",
        "anytime-valid risk process",
        "Exact oracle rejection is authoritative",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5639_gates_and_preregistration_are_frozen(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5639-GATES: immutable gates and thresholds precede outcomes."""

    assert mod.validate_artifact(artifact) is True
    gates = artifact["upstream_gate_receipts"]
    thresholds = artifact["preregistered_thresholds"]

    assert gates["both_structured_gates_enforced"] is True
    assert gates["exp5628"]["continuous_self_learning_ready"] is True
    assert gates["exp5638"]["gate_contract_ready_score"] == 1.0
    assert gates["exp5638"]["unsafe_false_accept_count_total"] == 0
    assert thresholds["frozen_before_heldout_outcomes"] is True
    assert thresholds["alpha"] == pytest.approx(mod.ALPHA)
    assert thresholds["delta"] == pytest.approx(mod.DELTA)
    assert thresholds["risk_limit"] == pytest.approx(mod.RISK_LIMIT)
    assert thresholds["stopping_time_schedule"] == list(mod.DEFAULT_STOPPING_TIMES)
    assert thresholds["fixed_nonoracle_arms"] == list(mod.FIXED_NONORACLE_ARMS)

    assert artifact["immutable_inputs"]["exp5628"]["path"] == (
        mod.EXP5628_ARTIFACT_RELATIVE_PATH.as_posix()
    )
    assert artifact["input_hashes"]["exp5628"].startswith("sha256:")
    assert artifact["input_hashes"]["exp5638"].startswith("sha256:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_scenario_learn_5639_recomputes_metrics_from_rows_and_ledgers(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5639-RECOMPUTE: audit metrics are not copied aggregates."""

    recomputation = artifact["independent_metric_recomputation"]
    assert recomputation["exp5628_aggregate_metrics_used_as_authority"] is False
    assert recomputation["row_level_replay_performed"] is True
    assert recomputation["decision_ledger_rows_replayed"] > 0
    assert recomputation["conformal_prediction_rows_replayed"] > 0

    full = mod.FULL_CONFORMAL_KAN_ARM
    full_ale = artifact["ale_by_arm"][full]["mean"]
    for arm in mod.FIXED_NONORACLE_ARMS:
        assert artifact["ale_by_arm"][arm]["mean"] > full_ale
        interval = artifact["paired_benefit_intervals"][arm]
        assert interval["lower"] > mod.PAIRED_BENEFIT_LOWER_FLOOR
        assert interval["n"] >= len(mod.exp5616.condition_keys())

    assert artifact["conditional_regret_by_group"]["bounded"] is True
    assert artifact["marginal_coverage"]["coverage"] >= mod.COVERAGE_FLOOR
    assert artifact["worst_group_coverage"]["coverage"] >= mod.COVERAGE_FLOOR
    assert artifact["worst_group_coverage"]["adequately_powered_groups_only"] is True
    assert artifact["unsafe_false_accept_count_total"] == 0
    assert artifact["retention_pass"] is True
    assert artifact["poison_rejection_pass"] is True
    assert artifact["checkpoint_replay_pass"] is True
    assert artifact["fr11_independent_promotion_ready_score"] == 1.0
    assert artifact["critical_flag_count"] == 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_learn_5639_anytime_pathwise_certificate_blocks_bad_prefix() -> None:
    """SCENARIO-LEARN-5639-ANYTIME: stops are registered and unsafe prefixes fail."""

    thresholds = mod.preregistered_thresholds()
    safe_outcomes = [False] * max(thresholds["stopping_time_schedule"])
    process = mod.anytime_risk_process(safe_outcomes, thresholds)

    assert [row["stop"] for row in process] == thresholds["stopping_time_schedule"]
    assert all(row["unsafe_count"] == 0 for row in process)
    assert all(row["upper_bound"] <= thresholds["risk_limit"] for row in process)
    assert mod.pathwise_risk_pass(process, thresholds) is True

    unsafe_outcomes = list(safe_outcomes)
    unsafe_outcomes[mod.DEFAULT_STOPPING_TIMES[0] - 1] = True
    unsafe_process = mod.anytime_risk_process(unsafe_outcomes, thresholds)
    assert unsafe_process[0]["unsafe_count"] == 1
    assert mod.pathwise_risk_pass(unsafe_process, thresholds) is False
    assert mod.exact_unsafe_false_accept({"exact_valid": False, "action_set": ["adapt"]}) is True
    assert mod.exact_unsafe_false_accept({"exact_valid": False, "action_set": ["abstain"]}) is False


def test_scenario_learn_5639_adversarial_controls_and_validation_fail_closed(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5639-ADVERSARIAL: stressed failures block promotion."""

    controls = artifact["adversarial_controls"]
    expected_controls = {
        "unseen_family_groups",
        "delayed_labels",
        "order_preserving_block_permutation",
        "prefix_stopping",
        "checkpoint_restart",
        "poison",
        "inactive_spline_substitution",
        "conformal_layer_disablement",
        "corrupted_row_artifact",
        "corrupted_control_artifact",
    }
    assert expected_controls == set(controls)
    assert all(control["pass"] is True for control in controls.values())

    bad_pathwise = deepcopy(artifact)
    bad_pathwise["pathwise_risk_upper_bound"][0]["upper_bound"] = (
        bad_pathwise["preregistered_thresholds"]["risk_limit"] + 0.01
    )
    bad_pathwise["fr11_independent_promotion_ready_score"] = mod.promotion_ready_score(
        bad_pathwise
    )
    bad_pathwise["honest_verdict"] = mod.honest_verdict(bad_pathwise)
    bad_pathwise["reproducibility_checksum"] = mod.reproducibility_checksum(bad_pathwise)
    with pytest.raises(ValueError, match="pathwise_risk_upper_bound"):
        mod.validate_artifact(bad_pathwise)

    bad_control = deepcopy(artifact)
    bad_control["adversarial_controls"]["poison"]["pass"] = False
    bad_control["critical_flag_count"] = mod.critical_flag_count(bad_control)
    bad_control["fr11_independent_promotion_ready_score"] = mod.promotion_ready_score(
        bad_control
    )
    bad_control["honest_verdict"] = mod.honest_verdict(bad_control)
    bad_control["reproducibility_checksum"] = mod.reproducibility_checksum(bad_control)
    with pytest.raises(ValueError, match="adversarial_controls"):
        mod.validate_artifact(bad_control)


def test_req_learn_5639_artifact_write_and_corrupt_inputs_rejected(
    tmp_path: Path,
    artifact: dict[str, object],
) -> None:
    """REQ-LEARN-5639: output is stable and corrupted artifacts fail closed."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == written
    assert mod.validate_artifact(written) is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in written
        assert written["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    with pytest.raises(ValueError, match="duplicate JSON key"):
        mod.load_json_object_from_bytes(b'{"unsafe": 0, "unsafe": 1}')
    with pytest.raises(ValueError, match="source artifact must be a JSON object"):
        mod.load_json_object_from_bytes(b"[]")
    with pytest.raises(ValueError, match="row_id"):
        mod.validate_control_row({"accepted_by_exact_validator": False})

    bad_score = deepcopy(artifact)
    bad_score["fr11_independent_promotion_ready_score"] = 0.0
    bad_score["honest_verdict"] = mod.honest_verdict(bad_score)
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="fr11_independent_promotion_ready_score"):
        mod.validate_artifact(bad_score)


def test_req_learn_5639_helper_and_validation_error_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: dict[str, object],
) -> None:
    """REQ-LEARN-5639: edge cases are deterministic and fail closed."""

    assert mod._display_path(REPO, REPO / "results") == "results"
    assert mod._display_path(REPO, tmp_path / "outside") == (tmp_path / "outside").as_posix()
    assert (
        mod.validate_control_row(
            {"row_id": "control-1", "accepted_by_exact_validator": False}
        )
        is True
    )
    assert mod.binomial_cdf_at_most(5, 5, 0.5) == 1.0
    assert mod.binomial_cdf_at_most(0, 5, 0.0) == 1.0
    assert mod.binomial_cdf_at_most(0, 5, 1.0) == 0.0
    with pytest.raises(ValueError, match="n must be positive"):
        mod.clopper_pearson_upper(0, 0, 0.01)
    assert mod.clopper_pearson_upper(5, 5, 0.01) == 1.0

    short_thresholds = mod.preregistered_thresholds()
    with pytest.raises(ValueError, match="stopping time"):
        mod.anytime_risk_process([False] * 10, short_thresholds)
    assert mod.critical_flag_count({"adversarial_controls": []}) == 1

    monkeypatch.setattr(
        mod,
        "upstream_gate_receipts",
        lambda _root: {"both_structured_gates_enforced": False},
    )
    with pytest.raises(ValueError, match="upstream structured gates"):
        mod.build_artifact(
            root=REPO,
            tests_added_or_reused=TESTS_ADDED_OR_REUSED,
            checkpoint_dir=tmp_path / "blocked",
        )

    assert any("missing required fields" in error for error in mod.artifact_errors({}))

    bad_cases: list[tuple[str, dict[str, object]]] = []

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    bad_cases.append(("field_principles", bad))

    bad = deepcopy(artifact)
    bad["preregistered_thresholds"] = deepcopy(bad["preregistered_thresholds"])
    bad["preregistered_thresholds"]["alpha"] = 0.2
    bad_cases.append(("preregistered_thresholds", bad))

    for field, value, expected in (
        ("inference_substrate", "live_llm", "inference_substrate"),
        (
            "upstream_gate_receipts",
            {"both_structured_gates_enforced": False},
            "upstream_gate_receipts",
        ),
        ("input_hashes", {}, "input_hashes"),
        (
            "conditional_regret_by_group",
            {"bounded": False},
            "conditional_regret_by_group",
        ),
        ("marginal_coverage", {"coverage": 0.0}, "marginal_coverage"),
        ("worst_group_coverage", {"coverage": 0.0}, "worst_group_coverage"),
        ("stopping_time_schedule", [], "stopping_time_schedule"),
        ("unsafe_false_accept_count_total", 1, "unsafe_false_accept_count_total"),
        ("retention_pass", False, "retention_pass"),
        ("poison_rejection_pass", False, "poison_rejection_pass"),
        ("checkpoint_replay_pass", False, "checkpoint_replay_pass"),
        ("adversarial_controls", [], "adversarial_controls"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad_cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["immutable_inputs"]["exp5628"]["read_only"] = False
    bad_cases.append(("immutable_inputs", bad))

    bad = deepcopy(artifact)
    bad["independent_metric_recomputation"][
        "exp5628_aggregate_metrics_used_as_authority"
    ] = True
    bad_cases.append(("independent_metric_recomputation", bad))

    bad = deepcopy(artifact)
    first_fixed = mod.FIXED_NONORACLE_ARMS[0]
    bad["paired_benefit_intervals"][first_fixed]["lower"] = 0.0
    bad_cases.append(("paired_benefit_intervals", bad))

    bad = deepcopy(artifact)
    bad["critical_flag_count"] = 99
    bad_cases.append(("critical_flag_count", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "ambiguous"
    bad_cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: stale"
    bad_cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    bad_cases.append(("reproducibility_checksum", bad))

    for expected, bad_artifact in bad_cases:
        assert any(expected in error for error in mod.artifact_errors(bad_artifact)), expected
