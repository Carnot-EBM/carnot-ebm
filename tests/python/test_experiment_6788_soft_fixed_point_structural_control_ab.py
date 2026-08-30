"""Tests for the paired grouped-versus-flat fixed-point comparison.

Spec refs: REQ-VERIFY-6788 and SCENARIO-VERIFY-6788-*.
"""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from carnot import experiment_6786_constraint_dependency_hard_negative_fixture as fixture
from carnot import experiment_6787_group_aware_soft_fixed_point as grouped_source
from carnot import experiment_6788_soft_fixed_point_structural_control_ab as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"
SOURCE_6786 = REPO_ROOT / exp.SOURCE_6786_RELATIVE_PATH
SOURCE_6787 = REPO_ROOT / exp.SOURCE_6787_RELATIVE_PATH


@pytest.fixture(scope="module")
def sources() -> tuple[dict, dict]:
    """Load immutable source artifacts once to reduce focused-test time."""

    return exp.load_json_object(SOURCE_6786), exp.load_json_object(SOURCE_6787)


@pytest.fixture(scope="module")
def context(sources: tuple[dict, dict]) -> exp.ExperimentContext:
    """Build the legal proposal projection and separate exact authority."""

    return exp.build_context(*sources)


def test_req_verify_6788_spec_declares_paired_control_contract() -> None:
    """REQ-VERIFY-6788 anchors isolation, pairing, resume, inference, and blocking."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-VERIFY-6788")
    section = spec[start : spec.index("### SCENARIO-VERIFY-6745", start)]
    for marker in (
        "REQ-VERIFY-6788",
        "SCENARIO-VERIFY-6788-ARM-ISOLATION",
        "SCENARIO-VERIFY-6788-PAIRED-BUDGETS",
        "SCENARIO-VERIFY-6788-EXACT-SEPARATION",
        "SCENARIO-VERIFY-6788-CHECKPOINT-RESUME",
        "SCENARIO-VERIFY-6788-PAIRED-INFERENCE",
        "SCENARIO-VERIFY-6788-BLOCKED",
        "complete_blocked_fixed_point_control_ab",
        "fixed_point_comparison_completed",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in exp.STANDARD_ARTIFACT_FIELDS


def test_req_verify_6788_preconditions_freeze_sources_splits_seeds_and_budget(
    sources: tuple[dict, dict], tmp_path: Path
) -> None:
    """REQ-VERIFY-6788 checks every authority before either arm fits."""

    summary = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        source_6786_path=SOURCE_6786,
        source_6787_path=SOURCE_6787,
        cpu_wall_budget_s=exp.CPU_WALL_BUDGET_S,
    )
    assert summary["all_passed"] is True
    assert summary["planned_row_count"] == exp.PLANNED_ROW_COUNT == 640
    assert summary["observed_seeds"] == exp.FROZEN_SEEDS
    assert summary["oracle_feature_violations"] == []
    assert {check["check"] for check in summary["checks"]} >= {
        "soft_fixed_point_proposer_ready",
        "exp6786_artifact_hash",
        "exp6787_artifact_hash",
        "source_artifact_hashes",
        "frozen_splits",
        "five_frozen_seeds",
        "legal_feature_contract",
        "planned_rows_fit_cpu_wall_budget",
    }

    blocked = exp.build_artifact(
        run_date="20260830",
        repo_root=tmp_path,
        source_6786_path=tmp_path / "missing-6786.json",
        source_6787_path=tmp_path / "missing-6787.json",
        checkpoint_path=tmp_path / "blocked-checkpoint.json",
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "complete_blocked_fixed_point_control_ab"
    assert blocked["rows"] == []
    assert blocked["fixed_point_comparison_completed"] is False
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_fixed_point_control_ab")
    assert blocked["gate_check_summary"]["first_failure"]["observed"] is False

    source_6786, source_6787 = sources
    drifted = deepcopy(source_6787)
    drifted["frozen_hyperparameters"]["seeds"] = [1, 2, 3, 4, 5]
    drifted_path = tmp_path / "drifted-6787.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    failed = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        source_6786_path=SOURCE_6786,
        source_6787_path=drifted_path,
        expected_6787_hash=exp.sha256_file(drifted_path),
    )
    assert failed["all_passed"] is False
    assert any(
        check["check"] == "five_frozen_seeds" and not check["passed"] for check in failed["checks"]
    )
    assert source_6786["constraint_group_fixture_ready"] is True


def test_scenario_verify_6788_flat_arm_isolated_and_parameter_matched(
    context: exp.ExperimentContext,
) -> None:
    """SCENARIO-VERIFY-6788-ARM-ISOLATION removes topology without reducing capacity."""

    seed = exp.FROZEN_SEEDS[0]
    models = exp.build_arm_models(seed)
    grouped = models[exp.GROUPED_ARM]
    flat = models[exp.FLAT_ARM]
    assert grouped is not flat
    assert exp.trainable_parameter_count(grouped) == 91
    assert exp.trainable_parameter_count(flat) == 91
    assert exp.parameter_match_fraction(models) == 0.0
    assert all(
        torch.equal(left, right)
        for left, right in zip(grouped.parameters(), flat.parameters(), strict=True)
    )

    unit = next(unit for unit in context.proposal_units if unit["split"] == "development")
    features = unit["proposal_features"]
    state = grouped_source.initial_variable_state(features, seed=seed)
    changed = deepcopy(features)
    changed["dependency_edges"] = []
    changed["topology_family"] = "changed-only-for-isolation-test"
    flat_original = flat.recurrent_step(state, features)
    flat_changed = flat.recurrent_step(state, changed)
    assert torch.equal(flat_original.variable_state, flat_changed.variable_state)
    assert flat_original.group_messages is None
    assert flat_original.dependency_messages is None
    grouped_original = exp.recurrent_step(grouped, state, features, arm=exp.GROUPED_ARM)
    grouped_changed = exp.recurrent_step(grouped, state, changed, arm=exp.GROUPED_ARM)
    assert not torch.equal(grouped_original.variable_state, grouped_changed.variable_state)

    with pytest.raises(ValueError, match="state shape"):
        flat.recurrent_step(torch.ones((1, 2), dtype=torch.float64), features)
    with pytest.raises(ValueError, match="unknown arm"):
        exp.recurrent_step(flat, state, features, arm="unknown")


def test_scenario_verify_6788_matched_training_and_seed_rotation(
    context: exp.ExperimentContext,
) -> None:
    """SCENARIO-VERIFY-6788-PAIRED-BUDGETS rotates equal seeds and update counts."""

    train_units = [unit for unit in context.proposal_units if unit["split"] == "train"]
    first_seed, second_seed = exp.FROZEN_SEEDS[:2]
    first_models = exp.build_arm_models(first_seed)
    repeated_models = exp.build_arm_models(first_seed)
    changed_models = exp.build_arm_models(second_seed)
    for arm in exp.ARMS:
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                first_models[arm].parameters(), repeated_models[arm].parameters(), strict=True
            )
        )
        assert any(
            not torch.equal(left, right)
            for left, right in zip(
                first_models[arm].parameters(), changed_models[arm].parameters(), strict=True
            )
        )

    receipts = {}
    for arm in exp.ARMS:
        model, receipt = exp.fit_arm(
            train_units,
            arm=arm,
            seed=first_seed,
            hyperparameters=exp.FROZEN_HYPERPARAMETERS,
        )
        receipts[arm] = receipt
        assert exp.trainable_parameter_count(model) == 91
        assert receipt["optimizer_update_count"] == exp.FROZEN_HYPERPARAMETERS["training_steps"]
        assert receipt["train_splits_seen"] == ["train"]
        assert len(receipt["loss_history"]) == exp.FROZEN_HYPERPARAMETERS["training_steps"]
    assert (
        receipts[exp.GROUPED_ARM]["optimizer_update_count"]
        == receipts[exp.FLAT_ARM]["optimizer_update_count"]
    )

    with pytest.raises(ValueError, match="train split"):
        exp.fit_arm(
            [next(unit for unit in context.proposal_units if unit["split"] == "development")],
            arm=exp.FLAT_ARM,
            seed=first_seed,
            hyperparameters=exp.FROZEN_HYPERPARAMETERS,
        )


def test_scenario_verify_6788_exact_checker_runs_after_raw_proposal(
    context: exp.ExperimentContext,
) -> None:
    """SCENARIO-VERIFY-6788-EXACT-SEPARATION appends labels without model feedback."""

    seed = exp.FROZEN_SEEDS[0]
    train_units = [unit for unit in context.proposal_units if unit["split"] == "train"]
    unit = next(unit for unit in context.proposal_units if unit["split"] == "held_topology_test")
    model, receipt = exp.fit_arm(
        train_units,
        arm=exp.FLAT_ARM,
        seed=seed,
        hyperparameters=exp.FROZEN_HYPERPARAMETERS,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    raw = exp.propose_raw_row(
        model,
        unit,
        arm=exp.FLAT_ARM,
        seed=seed,
        parameter_count=exp.trainable_parameter_count(model),
        optimizer_update_count=receipt["optimizer_update_count"],
        hyperparameters=exp.FROZEN_HYPERPARAMETERS,
    )
    assert raw["group_message_presence"] is False
    assert raw["group_message_receipts"] == []
    assert "exact_outcomes" not in raw
    assert all("exact_valid" not in candidate for candidate in raw["candidates"])

    scored = exp.attach_exact_outcomes(
        raw, context.exact_units[unit["unit_id"]], context.hard_negatives[unit["unit_id"]]
    )
    assert len(scored["exact_outcomes"]) == exp.FROZEN_HYPERPARAMETERS["candidate_count"]
    assert scored["candidate_hashes"] == raw["candidate_hashes"]
    assert scored["hard_negative_discrimination"]["positive_count"] > 0
    assert scored["hard_negative_discrimination"]["hard_negative_count"] == 1
    assert 0.0 <= scored["hard_negative_discrimination"]["auroc"] <= 1.0
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    for outcome in scored["exact_outcomes"]:
        checked = fixture.evaluate_candidate(
            context.exact_units[unit["unit_id"]], outcome["assignment"]
        )
        assert outcome["exact_valid"] == checked["exact_valid"]
        assert outcome["distance_to_nearest_valid"] >= 0


def test_scenario_verify_6788_checkpoint_resume_preserves_completed_cells(
    context: exp.ExperimentContext, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6788-CHECKPOINT-RESUME suppresses completed row payloads."""

    output_unit = next(unit for unit in context.proposal_units if unit["split"] == "development")
    seed = exp.FROZEN_SEEDS[0]
    manifest = exp.frozen_manifest([output_unit], seeds=[seed])
    checkpoint_path = tmp_path / "rows.json"

    first = exp.execute_cells(
        context=context,
        output_units=[output_unit],
        seeds=[seed],
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        stop_after_new_rows=1,
    )
    assert first["new_row_count"] == 1
    first_hash = first["rows"][0]["payload_hash"]
    first_bytes = checkpoint_path.read_bytes()

    second = exp.execute_cells(
        context=context,
        output_units=[output_unit],
        seeds=[seed],
        checkpoint_path=checkpoint_path,
        manifest=manifest,
    )
    assert second["new_row_count"] == 1
    assert len(second["rows"]) == 2
    assert second["rows"][0]["payload_hash"] == first_hash
    assert checkpoint_path.read_bytes() != first_bytes

    final_bytes = checkpoint_path.read_bytes()
    third = exp.execute_cells(
        context=context,
        output_units=[output_unit],
        seeds=[seed],
        checkpoint_path=checkpoint_path,
        manifest=manifest,
    )
    assert third["new_row_count"] == 0
    assert checkpoint_path.read_bytes() == final_bytes
    assert {row["payload"]["arm"] for row in third["rows"]} == set(exp.ARMS)


def test_scenario_verify_6788_paired_bootstrap_is_unit_clustered_and_deterministic(
    context: exp.ExperimentContext, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6788-PAIRED-INFERENCE keeps unit and arm pairs together."""

    output_units = [
        next(unit for unit in context.proposal_units if unit["split"] == split)
        for split in grouped_source.OUTPUT_SPLITS
    ]
    manifest = exp.frozen_manifest(output_units, seeds=[exp.FROZEN_SEEDS[0]])
    execution = exp.execute_cells(
        context=context,
        output_units=output_units,
        seeds=[exp.FROZEN_SEEDS[0]],
        checkpoint_path=tmp_path / "small-rows.json",
        manifest=manifest,
    )
    rows = [envelope["payload"] for envelope in execution["rows"]]
    first = exp.aggregate_rows(rows, bootstrap_resamples=100, bootstrap_seed=123)
    second = exp.aggregate_rows(rows, bootstrap_resamples=100, bootstrap_seed=123)
    assert first == second
    assert first["paired_exact_valid_delta_ci95"]["resamples"] == 100
    assert first["paired_exact_valid_delta_ci95"]["resampling_unit"] == (
        "unit_inside_topology_family"
    )
    assert set(first["metrics_by_topology"]) == {
        "directed_implication_star",
        "directed_implication_cycle",
    }
    assert first["paired_key_count"] == 2
    assert exp.row_attribution_errors(rows, manifest) == []

    broken = rows[:-1]
    assert any("paired arms" in error for error in exp.row_attribution_errors(broken, manifest))


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the complete 640-cell comparison once for artifact-level assertions."""

    root = tmp_path_factory.mktemp("exp6788-complete")
    return exp.build_artifact(
        run_date="20260830",
        repo_root=REPO_ROOT,
        source_6786_path=SOURCE_6786,
        source_6787_path=SOURCE_6787,
        checkpoint_path=root / "checkpoint.json",
        duration_s=10.0,
    )


def test_req_verify_6788_complete_artifact_is_row_derived(complete_artifact: dict) -> None:
    """REQ-VERIFY-6788 emits every required field and all attributable paired cells."""

    artifact = complete_artifact
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["fixed_point_comparison_completed"] is True
    assert artifact["cold_recompute_agreement"]["agreement"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["rows"]) == exp.PLANNED_ROW_COUNT
    assert artifact["parameter_counts_by_arm"] == {arm: 91 for arm in exp.ARMS}
    assert artifact["optimization_steps_by_arm"] == {arm: 30 for arm in exp.ARMS}
    assert artifact["candidate_budget_by_arm"] == {arm: 960 for arm in exp.ARMS}
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.headline_consistency_errors(artifact) == []

    pairs: dict[str, set[str]] = {}
    for row in artifact["rows"]:
        pairs.setdefault(row["paired_key"], set()).add(row["arm"])
        assert len(row["candidates"]) == 3
        assert len(row["raw_candidate_vectors"]) == 3
        assert len(row["exact_outcomes"]) == 3
        assert row["parameter_count"] == 91
        assert row["optimizer_update_count"] == 6
    assert len(pairs) == 320
    assert all(arms == set(exp.ARMS) for arms in pairs.values())


def test_req_verify_6788_validation_detects_headline_and_schema_drift(
    complete_artifact: dict,
) -> None:
    """REQ-VERIFY-6788 refuses row-to-headline, enum, and terminal-prefix drift."""

    mutations = (
        (lambda value: value.pop("schema"), "required field set mismatch"),
        (
            lambda value: value["field_principles"].pop("rows"),
            "field principle coverage mismatch",
        ),
        (
            lambda value: value.__setitem__("inference_substrate", "bad"),
            "inference substrate mismatch",
        ),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s must be non-negative"),
        (lambda value: value.__setitem__("random_seed", -1), "random seed mismatch"),
        (lambda value: value.__setitem__("verdict_class", "bad"), "verdict class is outside"),
        (lambda value: value.__setitem__("honest_verdict", "bad"), "terminal prefix"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (
            lambda value: value.__setitem__(
                "paired_exact_valid_delta", value["paired_exact_valid_delta"] + 0.25
            ),
            "headline metrics do not match rows",
        ),
        (lambda value: value.__setitem__("rows", value["rows"][:-1]), "row attribution"),
        (
            lambda value: value.__setitem__("fixed_point_comparison_completed", False),
            "complete artifact completion flag mismatch",
        ),
        (
            lambda value: value.__setitem__("reproducibility_checksum", "bad"),
            "reproducibility checksum mismatch",
        ),
    )
    for mutate, expected_error in mutations:
        changed = deepcopy(complete_artifact)
        mutate(changed)
        assert any(expected_error in error for error in exp.validate_artifact(changed))


def test_req_verify_6788_statistics_and_worker_boundaries_are_explicit(
    complete_artifact: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6788 handles ties, empty classes, cold workers, and invalid dates."""

    assert exp.binary_auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert exp.binary_auroc([0.5], [0.5]) == 0.5
    assert exp.binary_auroc([], [0.5]) is None
    assert exp.percentile([1.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="nonempty"):
        exp.percentile([], 0.5)
    with pytest.raises(ValueError, match="quantile"):
        exp.percentile([1.0], 2.0)
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(run_date="2026-08-30")

    payload = exp.cold_recompute_payload(complete_artifact["rows"])
    monkeypatch.setattr(exp.sys, "stdin", StringIO(json.dumps(payload)))
    assert exp._cold_recompute_worker() == 0
    worker = json.loads(capsys.readouterr().out)
    assert worker["aggregate_hash"] == exp.sha256_json(worker["aggregates"])

    monkeypatch.setattr(exp, "_cold_recompute_worker", lambda: 7)
    assert exp.main(["--cold-recompute-worker"]) == 7

    output = tmp_path / "cli-blocked.json"
    checkpoint = tmp_path / "cli-checkpoint.json"
    assert (
        exp.main(
            [
                "--date",
                "20260830",
                "--source-6786",
                str(tmp_path / "missing-6786.json"),
                "--source-6787",
                str(tmp_path / "missing-6787.json"),
                "--artifact-path",
                str(output),
                "--checkpoint-path",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert "complete_blocked_fixed_point_control_ab" in capsys.readouterr().out

    relative = exp.write_outputs(
        run_date="20260830",
        repo_root=tmp_path,
        source_6786_path=tmp_path / "missing-6786.json",
        source_6787_path=tmp_path / "missing-6787.json",
        artifact_path=Path("nested/blocked.json"),
        checkpoint_path=Path("nested/checkpoint.json"),
    )
    assert relative["verdict_class"] == "blocked"
    assert (tmp_path / "nested/blocked.json").is_file()


def test_req_verify_6788_defensive_branches_fail_closed(
    sources: tuple[dict, dict],
    context: exp.ExperimentContext,
    complete_artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6788 tests malformed authority, rows, recurrence, and cold replay."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        exp.load_json_object(scalar)
    with pytest.raises(ValueError, match="hidden_width"):
        exp.FlatRecurrentControl(hidden_width=0, seed=1)
    with pytest.raises(ValueError, match="unknown arm"):
        exp._model_for_arm("unknown", 1)

    source_6786, source_6787 = sources
    with monkeypatch.context() as patch:
        patch.setattr(grouped_source, "audit_feature_contract", lambda _units: ["denied"])
        with pytest.raises(ValueError, match="oracle feature refusal"):
            exp.build_context(source_6786, source_6787)
        train_units = [unit for unit in context.proposal_units if unit["split"] == "train"]
        with pytest.raises(ValueError, match="oracle feature refusal"):
            exp.fit_arm(
                train_units,
                arm=exp.FLAT_ARM,
                seed=exp.FROZEN_SEEDS[0],
                hyperparameters=exp.FROZEN_HYPERPARAMETERS,
            )
        model = exp.FlatRecurrentControl(hidden_width=8, seed=1)
        with pytest.raises(ValueError, match="oracle feature refusal"):
            exp.propose_raw_row(
                model,
                context.proposal_units[0],
                arm=exp.FLAT_ARM,
                seed=1,
                parameter_count=91,
                optimizer_update_count=6,
                hyperparameters=exp.FROZEN_HYPERPARAMETERS,
            )

    missing_unit = deepcopy(source_6786)
    missing_unit["rows"] = missing_unit["rows"][1:]
    with pytest.raises(ValueError, match="unit IDs differ"):
        exp.build_context(missing_unit, source_6787)
    bad_allowlist = deepcopy(source_6787)
    bad_allowlist["feature_allowlist"] = []
    with pytest.raises(ValueError, match="allowlist drifted"):
        exp.build_context(source_6786, bad_allowlist)

    unit = next(unit for unit in context.proposal_units if unit["split"] == "development")
    model = exp.FlatRecurrentControl(hidden_width=8, seed=1)
    with pytest.raises(ValueError, match="iteration_cap"):
        exp.run_arm_fixed_point(
            model,
            unit,
            arm=exp.FLAT_ARM,
            seed=1,
            iteration_cap=0,
            convergence_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="convergence_tolerance"):
        exp.run_arm_fixed_point(
            model,
            unit,
            arm=exp.FLAT_ARM,
            seed=1,
            iteration_cap=1,
            convergence_tolerance=-1.0,
        )
    converged = exp.run_arm_fixed_point(
        model,
        unit,
        arm=exp.FLAT_ARM,
        seed=1,
        iteration_cap=2,
        convergence_tolerance=1.0,
    )
    assert converged["stop_reason"] == "converged"

    group_count = len(unit["group_ids"])
    with monkeypatch.context() as patch:
        patch.setattr(
            model,
            "recurrent_step",
            lambda _state, _features: exp.ArmStep(
                torch.full((group_count, 2), float("nan"), dtype=torch.float64),
                None,
                None,
                None,
            ),
        )
        nonfinite = exp.run_arm_fixed_point(
            model,
            unit,
            arm=exp.FLAT_ARM,
            seed=1,
            iteration_cap=2,
            convergence_tolerance=0.0,
        )
    assert nonfinite["stop_reason"] == "non_finite"

    exact_unit = context.exact_units[unit["unit_id"]]
    invalid_assignment = {variable: 0 for variable in unit["variable_ids"]}
    assert (
        exp._selected_group_state(exact_unit["graph"]["local_groups"][0], invalid_assignment)
        is None
    )
    no_valid = {**exact_unit, "exact_assignments": []}
    assert exp._nearest_valid_distance(no_valid, invalid_assignment) is None

    plain = {"row_id": "plain"}
    assert exp._decode_checkpoint_payload(plain) == plain
    encoded = exp._encode_checkpoint_payload(plain)
    encoded["row_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="payload hash mismatch"):
        exp._decode_checkpoint_payload(encoded)

    sample_rows = complete_artifact["rows"][:2]
    sample_unit = next(
        item for item in context.proposal_units if item["unit_id"] == sample_rows[0]["unit_id"]
    )
    sample_manifest = exp.frozen_manifest([sample_unit], seeds=[sample_rows[0]["random_seed"]])
    duplicate = [sample_rows[0], sample_rows[0]]
    assert "duplicate row IDs" in exp.row_attribution_errors(duplicate, sample_manifest)
    wrong_identity = deepcopy(sample_rows)
    wrong_identity[0]["row_id"] = "wrong"
    wrong_identity[0]["paired_key"] = "wrong-pair"
    wrong_identity[0]["candidate_budget"] = 99
    attribution = exp.row_attribution_errors(wrong_identity, sample_manifest)
    assert any("identity mismatch" in error for error in attribution)
    assert any("candidate budget mismatch" in error for error in attribution)
    assert any("paired keys" in error for error in attribution)
    assert exp.percentile([2.0], 0.5) == 2.0
    assert exp.headline_consistency_errors({"rows": []}) == []

    with monkeypatch.context() as patch:
        patch.setattr(
            exp.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=2, stderr="forced cold failure"),
        )
        with pytest.raises(RuntimeError, match="forced cold failure"):
            exp.run_cold_recompute(sample_rows, repo_root=REPO_ROOT)

    blocked = exp.build_artifact(
        repo_root=tmp_path,
        source_6786_path=tmp_path / "missing-6786.json",
        source_6787_path=tmp_path / "missing-6787.json",
        checkpoint_path=tmp_path / "unused.json",
        duration_s=0.1,
    )
    blocked_mutations = (
        (lambda value: value.__setitem__("status", "bad"), "blocked artifact status mismatch"),
        (lambda value: value.__setitem__("rows", [{}]), "blocked artifact must not contain rows"),
        (
            lambda value: value.__setitem__("fixed_point_comparison_completed", True),
            "blocked artifact cannot be complete",
        ),
        (
            lambda value: value["gate_check_summary"].__setitem__("all_passed", True),
            "blocked artifact must name failed preconditions",
        ),
    )
    for mutate, message in blocked_mutations:
        changed = deepcopy(blocked)
        mutate(changed)
        assert message in exp.validate_artifact(changed)

    complete_mutations = (
        (
            lambda value: value["gate_check_summary"].__setitem__("all_passed", False),
            "complete artifact has failed preconditions",
        ),
        (
            lambda value: value["cold_recompute_agreement"].__setitem__("agreement", False),
            "complete artifact lacks cold recompute agreement",
        ),
        (
            lambda value: value["parameter_counts_by_arm"].__setitem__(exp.FLAT_ARM, 1),
            "parameter counts do not match frozen architectures",
        ),
        (
            lambda value: value.__setitem__(
                "verdict_class",
                "null" if value["verdict_class"] == "positive" else "positive",
            ),
            "positive verdict does not match decision gates",
        ),
    )
    for mutate, message in complete_mutations:
        changed = deepcopy(complete_artifact)
        mutate(changed)
        assert any(message in error for error in exp.validate_artifact(changed))
