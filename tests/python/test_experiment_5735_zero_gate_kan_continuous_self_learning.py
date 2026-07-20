"""Tests for Exp5735 zero-gated KAN residual growth.

Spec refs: REQ-LEARN-5735,
SCENARIO-LEARN-5735-ZERO-GATE,
SCENARIO-LEARN-5735-CHRONOLOGY,
SCENARIO-LEARN-5735-BASELINES,
SCENARIO-LEARN-5735-RELEASE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5735_zero_gate_kan_continuous_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5735_zero_gate_kan_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5735_zero_gate_kan_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_5735_zero_gate_kan_continuous_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5735_zero_gate_kan_continuous_self_learning.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5735_zero_gate_kan_continuous_self_learning.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-LEARN-5735: build the zero-gate artifact once for schema tests."""

    base = tmp_path_factory.mktemp("exp5735")
    return mod.run(
        root=REPO,
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        ledger_path=base / mod.LEDGER_RELATIVE_PATH.name,
        checkpoint_dir=base / "checkpoints",
        test_commands=TEST_COMMANDS,
        write=True,
    )


def test_req_learn_5735_spec_declares_zero_gate_contract() -> None:
    """REQ-LEARN-5735: OpenSpec anchors fields, controls, and gate scalar."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5735") : spec.index("## REQ-LEARN-5640")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5735",
        "SCENARIO-LEARN-5735-ZERO-GATE",
        "SCENARIO-LEARN-5735-CHRONOLOGY",
        "SCENARIO-LEARN-5735-BASELINES",
        "SCENARIO-LEARN-5735-RELEASE",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`function_preserving_insertion_score` SHALL be exactly `1.0`",
        "no-growth active spline",
        "always-open residual",
        "parameter-matched MLP residual",
        "frozen controller",
        "corrupted-order controls",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5735_zero_gate_equivalence_and_ledger(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5735-ZERO-GATE: insertion preserves protected-prefix output."""

    assert mod.validate_artifact(artifact) is True
    assert artifact["function_preserving_insertion_score"] == 1.0
    assert artifact["pre_insertion_output_hash"] == artifact["post_insertion_output_hash"]
    assert all(row["passed"] is True for row in artifact["insertion_equivalence_receipts"])
    assert artifact["zero_gate_definition"]["initial_gate_scalar"] == 0.0
    assert artifact["zero_gate_definition"]["prefix_basis_is_zero"] is True
    assert artifact["gate_trajectory"][0]["phase"] == "insertion"
    assert artifact["gate_trajectory"][0]["gate"] == 0.0
    assert max(row["gate"] for row in artifact["gate_trajectory"]) > 0.0

    ledger_path = Path(str(artifact["operation_ledger_path"]))
    rows = mod.load_operation_ledger(ledger_path)
    assert len(rows) == artifact["exact_label_receipts"]["headline_prediction_count"]
    assert mod.verify_operation_ledger(rows, artifact) is True
    assert all(row["ledger_hash"] == mod.operation_ledger_row_hash(row) for row in rows)
    assert all(
        required in rows[0]
        for required in (
            "pre_label_prediction",
            "exact_label_receipt",
            "update_decision",
            "gate_before",
            "gate_after",
            "parameter_hash_before",
            "parameter_hash_after",
            "post_update_protected_prefix_check",
        )
    )
    assert any(row["phase"] == "suffix_residual_growth" for row in rows)
    assert all(
        row["post_update_protected_prefix_check"]["passed"] is True
        for row in rows
        if row["phase"] == "suffix_residual_growth"
    )


def test_scenario_learn_5735_chronology_exact_labels_and_baselines(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5735-BASELINES: controls share streams and seeds."""

    assert artifact["session_count"] == mod.SESSION_COUNT
    assert len(artifact["random_seeds"]) == mod.SESSION_COUNT
    assert len(set(artifact["random_seeds"])) == mod.SESSION_COUNT
    assert artifact["exact_label_receipts"]["label_error_count"] == 0
    assert artifact["exact_label_receipts"]["chronological_order_preserved"] is True
    assert artifact["stream_root_hash"].startswith("sha256:")
    assert artifact["stream_order_hash"].startswith("sha256:")
    assert all(row["available"] is True for row in artifact["preconditions_checked"].values())
    assert set(artifact["arm_configs"]) == set(mod.ARM_NAMES)
    assert artifact["arm_configs"][mod.MLP_RESIDUAL_ARM]["sidecar_only"] is True
    assert artifact["arm_configs"][mod.MLP_RESIDUAL_ARM]["parameter_count"] == (
        artifact["arm_configs"][mod.ZERO_GATED_ARM]["residual_parameter_count"]
    )

    metrics = artifact["arm_metrics"]
    assert metrics[mod.ZERO_GATED_ARM]["session_count"] == mod.SESSION_COUNT
    assert metrics[mod.NO_GROWTH_ARM]["session_count"] == mod.SESSION_COUNT
    assert artifact["suffix_improvement"] > 0.0
    assert metrics[mod.ZERO_GATED_ARM]["suffix_error"] < metrics[mod.NO_GROWTH_ARM]["suffix_error"]
    assert artifact["prefix_retention_delta"] <= mod.OLD_PREFIX_RETENTION_MARGIN
    assert artifact["unsafe_update_count"] == 0
    assert metrics[mod.CORRUPTED_ORDER_ARM]["chronological_order_preserved"] is False
    assert artifact["adversarial_controls"]["corrupted_order"]["detected"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_learn_5735_release_certificate_restart_and_costs(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5735-RELEASE: statistical and replay gates pass cleanly."""

    certificate = artifact["statistical_model_check_receipt"]
    assert certificate["session_count"] >= 30
    assert certificate["positive_session_count"] >= certificate["minimum_positive_sessions"]
    assert certificate["passes"] is True
    assert certificate["delta"] == pytest.approx(mod.DELTA)
    assert artifact["parameter_growth"] <= mod.MAX_PARAMETER_GROWTH
    assert artifact["peak_memory_growth_mb"] <= mod.MAX_MEMORY_GROWTH_MB
    assert artifact["update_latency_distribution"]["max"] <= mod.MAX_UPDATE_LATENCY_MS
    assert artifact["checkpoint_hashes"]["all_replay_exact"] is True
    assert artifact["restart_equivalence"]["passed"] is True
    assert artifact["rollback_receipt"]["passed"] is True
    assert artifact["zero_gate_csl_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["test_commands"] == TEST_COMMANDS
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    for field in artifact:
        assert field in artifact["field_principles"]


def test_req_learn_5735_run_writes_stable_artifact(artifact: dict[str, object], tmp_path: Path) -> None:
    """REQ-LEARN-5735: run output, ledger, and checksum replay exactly."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    ledger = tmp_path / mod.LEDGER_RELATIVE_PATH.name
    written = mod.run(
        root=REPO,
        result_path=destination,
        ledger_path=ledger,
        checkpoint_dir=tmp_path / "checkpoints",
        test_commands=TEST_COMMANDS,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == written
    assert mod.validate_artifact(written) is True
    assert Path(str(written["operation_ledger_path"])) == ledger
    assert mod.verify_operation_ledger(mod.load_operation_ledger(ledger), written) is True
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert written["stream_root_hash"] == artifact["stream_root_hash"]


def test_req_learn_5735_validation_fails_closed(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5735: bad gates, stale verdicts, and schema drift are rejected."""

    cases: list[tuple[str, dict[str, object]]] = []
    for field, value, expected in (
        ("function_preserving_insertion_score", 0.0, "function_preserving_insertion_score"),
        ("suffix_improvement", 0.0, "suffix_improvement"),
        ("prefix_retention_delta", mod.OLD_PREFIX_RETENTION_MARGIN + 0.1, "prefix_retention_delta"),
        ("unsafe_update_count", 1, "unsafe_update_count"),
        ("parameter_growth", mod.MAX_PARAMETER_GROWTH + 0.1, "parameter_growth"),
        ("peak_memory_growth_mb", mod.MAX_MEMORY_GROWTH_MB + 0.1, "peak_memory_growth_mb"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
        ("restart_equivalence", {"passed": False}, "restart_equivalence"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["statistical_model_check_receipt"]["passes"] = False
    cases.append(("statistical_model_check_receipt", bad))

    bad = deepcopy(artifact)
    bad["checkpoint_hashes"]["all_replay_exact"] = False
    cases.append(("checkpoint_hashes", bad))

    bad = deepcopy(artifact)
    bad["field_principles"].pop("suffix_improvement")
    cases.append(("field_principles", bad))

    bad = deepcopy(artifact)
    bad.pop("suffix_improvement")
    cases.append(("missing required fields", bad))

    bad = deepcopy(artifact)
    bad["zero_gate_csl_ready_score"] = 0.0
    cases.append(("zero_gate_csl_ready_score", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: stale"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    cases.append(("reproducibility_checksum", bad))

    for expected, bad_artifact in cases:
        if expected not in {"honest_verdict", "reproducibility_checksum", "zero_gate_csl_ready_score"}:
            bad_artifact["zero_gate_csl_ready_score"] = mod.zero_gate_csl_ready_score(bad_artifact)
            bad_artifact["honest_verdict"] = mod.honest_verdict(bad_artifact)
            bad_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(bad_artifact)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)


def test_req_learn_5735_helper_edges(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5735: helper edge cases remain deterministic and auditable."""

    rows, sessions = mod.select_chronological_sessions(REPO, session_count=mod.SESSION_COUNT)
    prefix_rows, suffix_rows = mod.protected_prefix_and_suffix(rows)
    state = mod.initial_sidecar_state(seed=mod.DEFAULT_RANDOM_SEEDS[0])
    pre = mod.output_vector(state, prefix_rows, prefix_length=len(prefix_rows))
    inserted = mod.insert_zero_gated_residual(state)
    post = mod.output_vector(inserted, prefix_rows, prefix_length=len(prefix_rows))

    assert sessions[0]["seed"] == mod.DEFAULT_RANDOM_SEEDS[0]
    assert mod.output_hash(pre) == mod.output_hash(post)
    assert mod.function_preserving_insertion_score(
        [{"passed": True}, {"passed": True}]
    ) == 1.0
    assert mod.function_preserving_insertion_score(
        [{"passed": True}, {"passed": False}]
    ) == 0.0
    assert mod.prefix_certificate(inserted, prefix_rows, pre, len(prefix_rows))["passed"] is True
    assert mod.prefix_certificate(
        mod.replace_state(inserted, gate=1.0, residual=mod.always_open_initial_residual()),
        prefix_rows,
        pre,
        len(prefix_rows),
    )["passed"] is False
    assert mod.binomial_upper_tail(30, 30, 0.5) < mod.DELTA
    assert mod.binomial_upper_tail(0, 30, 0.5) == pytest.approx(1.0)
    assert mod.latency_distribution([]) == {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert len(suffix_rows) > 0
    assert mod.artifact_errors({}) == ["missing required fields: " + str(list(mod.REQUIRED_ARTIFACT_FIELDS))]
    assert mod.verify_checkpoint_payloads(artifact["checkpoint_hashes"]["receipts"]) is True

    with pytest.raises(ValueError, match="session_count"):
        mod.load_selected_raw_rows(REPO, session_count=10_000)

    satisfied = mod.initial_sidecar_state(seed=mod.DEFAULT_RANDOM_SEEDS[0])
    satisfied.base += prefix_rows[0].label * prefix_rows[0].features * 100.0
    assert mod._apply_base_update(satisfied, prefix_rows[0]) == ("rejected_margin_satisfied", 0.0)

    bad_hash = deepcopy(artifact["checkpoint_hashes"]["receipts"])
    bad_hash[0]["checkpoint_hash"] = "sha256:bad"
    assert mod.verify_checkpoint_payloads(bad_hash) is False

    bad_state_hash = deepcopy(artifact["checkpoint_hashes"]["receipts"])
    bad_state_hash[0]["state_hash"] = "sha256:bad"
    assert mod.verify_checkpoint_payloads(bad_state_hash) is False

    non_mapping_principles = deepcopy(artifact)
    non_mapping_principles["field_principles"] = []
    non_mapping_principles["zero_gate_csl_ready_score"] = mod.zero_gate_csl_ready_score(
        non_mapping_principles
    )
    non_mapping_principles["honest_verdict"] = mod.honest_verdict(non_mapping_principles)
    non_mapping_principles["reproducibility_checksum"] = mod.reproducibility_checksum(
        non_mapping_principles
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(non_mapping_principles)
