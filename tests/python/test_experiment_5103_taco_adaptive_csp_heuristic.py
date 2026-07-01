"""Tests for Exp 5103 TACO-style adaptive CSP heuristic.

Spec refs: REQ-VERIFY-5103, SCENARIO-VERIFY-5103.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import pytest

from carnot import experiment_5103_taco_adaptive_csp_heuristic as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5103_spec_declares_adaptive_csp_contract() -> None:
    """REQ-VERIFY-5103: OpenSpec anchors the exact-solver advisory heuristic contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5103",
        "SCENARIO-VERIFY-5103",
        "python/carnot/experiment_5103_taco_adaptive_csp_heuristic.py",
        "results/experiment_5103_taco_adaptive_csp_heuristic_v468.json",
        mod.INFERENCE_SUBSTRATE,
        mod.SUCCESS_VERDICT,
        mod.NO_WIN_VERDICT,
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5103_family_has_exact_labels_across_splits() -> None:
    """SCENARIO-VERIFY-5103: generated train/dev/held-out CSP labels come from exact search."""

    family = mod.build_instance_family()
    solver = mod.ExactGraphColoringSolver()

    assert {instance.split for instance in family} == {"train", "dev", "heldout"}
    assert len(family) >= 6
    for instance in family:
        result = solver.solve(instance, mod.baseline_order(instance))
        assert instance.n_nodes <= mod.MAX_NODES
        assert result.status in {"colorable", "uncolorable"}
        assert result.colorable is instance.expected_colorable
        if result.colorable:
            assert result.assignment is not None
            assert solver.verify_assignment(instance, result.assignment)
        else:
            assert result.assignment is None

    first = family[0]
    assert not solver.verify_assignment(first, None)
    assert not solver.verify_assignment(first, [first.n_colors] * first.n_nodes)


def test_req_verify_5103_adapted_order_is_advisory_and_measurably_changes_effort() -> None:
    """REQ-VERIFY-5103: adaptation proposes order help while exact search still solves."""

    family = {instance.instance_id: instance for instance in mod.build_instance_family()}
    solver = mod.ExactGraphColoringSolver()
    hub = family["dev_hub_distractor_k4_8"]
    wheel = family["heldout_wheel8_even_sat"]

    hub_adaptation = mod.adapt_instance_order(hub)
    assert sorted(hub_adaptation.order) == list(range(hub.n_nodes))
    assert hub_adaptation.steps == mod.DEFAULT_ADAPTATION_STEPS
    assert hub_adaptation.heuristic_only_solution_counted is False

    hub_baseline = solver.solve(hub, mod.baseline_order(hub))
    hub_static = solver.solve(hub, mod.static_degree_order(hub))
    hub_adapted = solver.solve(hub, hub_adaptation.order)
    assert hub_adapted.effort_score < hub_baseline.effort_score
    assert hub_adapted.effort_score < hub_static.effort_score

    wheel_adaptation = mod.adapt_instance_order(wheel)
    wheel_baseline = solver.solve(wheel, mod.baseline_order(wheel))
    wheel_adapted = solver.solve(wheel, wheel_adaptation.order)
    assert wheel_adapted.effort_score > wheel_baseline.effort_score


def test_req_verify_5103_artifact_fields_principles_and_failure_cases() -> None:
    """REQ-VERIFY-5103: artifact compares all arms and reports harmful adapted cases."""

    artifact = mod.run(duration_s=0.0)

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "llm" not in artifact["inference_substrate"].lower()
    assert artifact["csp_family"] == mod.CSP_FAMILY
    assert artifact["exact_solver_backend"] == mod.EXACT_SOLVER_BACKEND
    assert artifact["correctness_preserved"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["instances_total"] == len(mod.build_instance_family())
    assert artifact["adaptation_steps"] == mod.DEFAULT_ADAPTATION_STEPS * artifact["instances_total"]
    assert artifact["harmful_instance_count"] > 0
    assert artifact["adapted_effort"]["total_effort_score"] < artifact["baseline_effort"]["total_effort_score"]
    assert artifact["delta_effort_vs_baseline"]["adapted"] < 0
    assert artifact["delta_effort_vs_baseline"]["static_heuristic"] < 0
    assert set(artifact["split_summaries"]) == {"train", "dev", "heldout"}

    for row in artifact["per_instance_results"]:
        assert row["heuristic_only_solution_counted"] is False
        exact_colorable = row["exact_label"]["colorable"]
        for arm_name in ("baseline", "static_heuristic", "adapted"):
            arm = row[arm_name]
            assert arm["colorable"] is exact_colorable
            if exact_colorable:
                assert arm["solution_verified"] is True


def test_scenario_verify_5103_write_artifact_round_trips(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5103: terminal JSON is written with the required schema."""

    artifact = mod.write_artifact(root=tmp_path)
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(payload)
    assert payload["result_path"] == mod.RESULT_RELATIVE_PATH
    assert payload["spec_refs"] == ["REQ-VERIFY-5103", "SCENARIO-VERIFY-5103"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("duration_s", -1.0, "duration_s"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("csp_family", "other", "csp_family"),
        ("exact_solver_backend", "heuristic_only", "exact_solver_backend"),
        ("correctness_preserved", False, "correctness_preserved"),
        ("harmful_instance_count", -1, "harmful_instance_count"),
        ("flagged_adversarial", True, "flagged_adversarial"),
    ],
)
def test_req_verify_5103_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5103: malformed terminal fields fail closed."""

    artifact = mod.run(duration_s=0.0)
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda artifact: artifact.pop("baseline_effort"), "missing required fields"),
        (lambda artifact: artifact.update({"field_principles": {}}), "field_principles"),
        (
            lambda artifact: artifact["delta_effort_vs_baseline"].update({"adapted": 0}),
            "delta_effort_vs_baseline",
        ),
        (
            lambda artifact: artifact["per_instance_results"][0].update({"heuristic_only_solution_counted": True}),
            "heuristic_only_solution_counted",
        ),
    ],
)
def test_req_verify_5103_validate_artifact_rejects_consistency_violations(
    mutator: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    """SCENARIO-VERIFY-5103: coherent-looking but dishonest artifacts fail closed."""

    artifact = mod.run(duration_s=0.0)
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5103_row_correctness_guards_cover_corrupt_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5103: exact-label and solution-verification drift fails closed."""

    artifact = mod.run(duration_s=0.0)
    uncolorable_row = json.loads(json.dumps(artifact["per_instance_results"][0]))
    colorable_row = next(
        json.loads(json.dumps(row))
        for row in artifact["per_instance_results"]
        if row["exact_label"]["colorable"]
    )

    wrong_expected = json.loads(json.dumps(uncolorable_row))
    wrong_expected["expected_colorable"] = not wrong_expected["expected_colorable"]
    assert mod._row_correctness_preserved(wrong_expected) is False

    wrong_arm_label = json.loads(json.dumps(uncolorable_row))
    wrong_arm_label["baseline"]["colorable"] = not wrong_arm_label["exact_label"]["colorable"]
    assert mod._row_correctness_preserved(wrong_arm_label) is False

    unverified_solution = json.loads(json.dumps(colorable_row))
    unverified_solution["adapted"]["solution_verified"] = False
    assert mod._row_correctness_preserved(unverified_solution) is False

    unsat_with_assignment = json.loads(json.dumps(uncolorable_row))
    unsat_with_assignment["adapted"]["assignment"] = [0] * unsat_with_assignment["n_nodes"]
    assert mod._row_correctness_preserved(unsat_with_assignment) is False

    monkeypatch.setattr(mod, "_row_correctness_preserved", lambda row: False)
    with pytest.raises(ValueError, match="correctness_preserved"):
        mod.run(duration_s=0.0)


def test_req_verify_5103_main_writes_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5103: CLI entrypoint writes the configured result path."""

    monkeypatch.setenv("CARNOT_EXP5103_ROOT", str(tmp_path))

    assert mod.main() == 0
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(payload)


def test_deliverable_file_validates_for_req_verify_5103() -> None:
    """SCENARIO-VERIFY-5103: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
