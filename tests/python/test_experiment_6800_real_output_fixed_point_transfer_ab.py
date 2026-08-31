"""Tests for frozen fixed-point transfer to authentic model-output graphs.

Spec refs: REQ-VERIFY-6800 and SCENARIO-VERIFY-6800-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from carnot import experiment_6800_real_output_fixed_point_transfer_ab as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load the five frozen sources once because they are immutable evidence."""

    return exp.load_sources(REPO_ROOT)


@pytest.fixture(scope="module")
def reconstruction(sources: dict[str, dict]) -> exp.FrozenReconstruction:
    """Reconstruct both arms once and prove all V592 hashes before transfer."""

    return exp.reconstruct_frozen_arms(sources)


def test_req_verify_6800_spec_declares_real_output_transfer_contract() -> None:
    """REQ-VERIFY-6800 anchors isolation, pairing, authority, resume, and blocks."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-VERIFY-6800")
    section = spec[start : spec.index("### SCENARIO-VERIFY-6745", start)]
    for marker in (
        "SCENARIO-VERIFY-6800-TRAIN-ISOLATION",
        "SCENARIO-VERIFY-6800-MATCHED-ARMS",
        "SCENARIO-VERIFY-6800-EXACT-AUTHORITY",
        "SCENARIO-VERIFY-6800-CLUSTERED-INFERENCE",
        "SCENARIO-VERIFY-6800-CHECKPOINT-RESUME",
        "SCENARIO-VERIFY-6800-DESTRUCTIVE-CONTROLS",
        "SCENARIO-VERIFY-6800-BLOCKED",
        "complete_blocked_real_output_fixed_point_transfer",
        "model_output_fixed_point_comparison_completed",
    ):
        assert marker in section
    for field in exp.TASK_REQUIRED_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_6800_preconditions_freeze_authority_grid_and_budget(
    sources: dict[str, dict], tmp_path: Path
) -> None:
    """REQ-VERIFY-6800 checks exact hashes, readiness, grid, seeds, and CPU budget."""

    summary = exp.evaluate_preconditions(repo_root=REPO_ROOT, sources=sources)
    assert summary["all_passed"] is True
    assert summary["grid_observation"] == {
        "case_count": 97,
        "transformation_count": 3,
        "source_models": [
            "gemma4_26b_middle_moe",
            "gemma4_31b_flagship_dense",
            "qwen36_flagship_moe",
        ],
        "constraint_families": [
            "expander_tseitin",
            "ladder_tseitin",
            "pigeonhole_anchor",
        ],
        "splits": {"development": 64, "held_case": 33},
        "planned_row_count": 2910,
    }
    assert summary["runtime_budget"]["planned_cpu_wall_budget_s"] == exp.CPU_WALL_BUDGET_S
    assert len(summary["source_artifact_hashes"]) == 5

    bad = deepcopy(sources)
    bad["exp6799"]["model_output_constraint_probe_ready"] = False
    blocked = exp.evaluate_preconditions(repo_root=REPO_ROOT, sources=bad)
    assert blocked["all_passed"] is False
    assert "model_output_constraint_probe_ready" in blocked["failed_checks"]

    missing = exp.evaluate_preconditions(repo_root=tmp_path)
    assert missing["all_passed"] is False
    assert missing["first_failure"]["check"] == "source_artifact:exp6786"


def test_scenario_verify_6800_reproduces_v592_before_freezing(
    reconstruction: exp.FrozenReconstruction,
) -> None:
    """SCENARIO-VERIFY-6800-MATCHED-ARMS reproduces every registered hash."""

    assert len(reconstruction.models) == 10
    assert len(reconstruction.training_isolation_receipts) == 10
    assert len(reconstruction.v592_reproduction_receipts) == 10
    assert (
        sum(row["candidate_row_count"] for row in reconstruction.v592_reproduction_receipts) == 640
    )
    assert all(row["agreement"] for row in reconstruction.v592_reproduction_receipts)
    assert all(
        row["train_splits_seen"] == ["train"] for row in reconstruction.training_isolation_receipts
    )
    assert all(
        row["training_source"] == "exp6786_frozen_train_split"
        for row in reconstruction.training_isolation_receipts
    )
    assert all(
        parameter.requires_grad is False
        for model in reconstruction.models.values()
        for parameter in model.parameters()
    )
    assert exp.parameter_counts(reconstruction.models) == {arm: 91 for arm in exp.ARMS}


def test_scenario_verify_6800_projection_and_exact_authority_are_isolated(
    sources: dict[str, dict], reconstruction: exp.FrozenReconstruction
) -> None:
    """SCENARIO-VERIFY-6800-EXACT-AUTHORITY appends checks after frozen hashes."""

    group = sources["exp6799"]["probe_groups"][0]
    unit = exp.project_transfer_unit(group, "restructuring")
    assert exp.audit_transfer_feature_contract([unit]) == []
    serialized = exp.canonical_json(unit["proposal_features"])
    for denied in exp.FEATURE_DENYLIST:
        assert denied not in serialized

    seed = exp.FROZEN_SEEDS[0]
    model = reconstruction.models[(seed, exp.GROUPED_ARM)]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    raw = exp.propose_transfer_row(model, unit, arm=exp.GROUPED_ARM, seed=seed)
    assert "exact_outcomes" not in raw
    hashes = deepcopy(raw["candidate_hashes"])
    scored = exp.attach_exact_outcomes(raw, unit["exact_graph_record"])
    assert scored["candidate_hashes"] == hashes
    assert scored["exact_evaluation_receipt"]["evaluated_after_proposal"] is True
    assert scored["exact_evaluation_receipt"]["model_feedback_applied"] is False
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    assert len(scored["exact_outcomes"]) == exp.CANDIDATE_COUNT
    assert all(outcome["distance_to_nearest_valid"] >= 0 for outcome in scored["exact_outcomes"])
    assert scored["hard_negative_control"]["local_checks_passed"] is True
    assert scored["hard_negative_control"]["exact_valid"] is False


def test_scenario_verify_6800_controls_change_only_declared_inputs(
    sources: dict[str, dict], reconstruction: exp.FrozenReconstruction
) -> None:
    """SCENARIO-VERIFY-6800-DESTRUCTIVE-CONTROLS keeps identity out of proposal."""

    group = sources["exp6799"]["probe_groups"][1]
    unit = exp.project_transfer_unit(group, "refinement")
    seed = exp.FROZEN_SEEDS[0]
    model = reconstruction.models[(seed, exp.GROUPED_ARM)]
    raw = exp.propose_transfer_row(model, unit, arm=exp.GROUPED_ARM, seed=seed)
    controls = raw["control_outcomes"]
    assert controls["source_model_identity_removal"]["proposal_input_unchanged"] is True
    assert controls["group_id_permutation"]["candidate_budget"] == exp.CANDIDATE_COUNT
    assert controls["dependency_edge_removal"]["candidate_budget"] == exp.CANDIDATE_COUNT
    assert controls["surface_relabeling"]["candidate_budget"] == exp.CANDIDATE_COUNT
    assert raw["candidate_budget"] == exp.CANDIDATE_COUNT


def test_scenario_verify_6800_checkpoint_resume_preserves_payloads(
    sources: dict[str, dict], reconstruction: exp.FrozenReconstruction, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6800-CHECKPOINT-RESUME executes only missing cells."""

    unit = exp.project_transfer_unit(sources["exp6799"]["probe_groups"][0], "base")
    manifest = exp.frozen_manifest([unit], seeds=[exp.FROZEN_SEEDS[0]])
    checkpoint = tmp_path / "rows.json"
    first = exp.execute_cells(
        units=[unit],
        reconstruction=reconstruction,
        checkpoint_path=checkpoint,
        manifest=manifest,
        seeds=[exp.FROZEN_SEEDS[0]],
        stop_after_new_rows=1,
    )
    assert first["new_row_count"] == 1
    first_hash = first["rows"][0]["payload_hash"]
    second = exp.execute_cells(
        units=[unit],
        reconstruction=reconstruction,
        checkpoint_path=checkpoint,
        manifest=manifest,
        seeds=[exp.FROZEN_SEEDS[0]],
    )
    assert second["new_row_count"] == 1
    assert second["rows"][0]["payload_hash"] == first_hash
    final_bytes = checkpoint.read_bytes()
    third = exp.execute_cells(
        units=[unit],
        reconstruction=reconstruction,
        checkpoint_path=checkpoint,
        manifest=manifest,
        seeds=[exp.FROZEN_SEEDS[0]],
    )
    assert third["new_row_count"] == 0
    assert checkpoint.read_bytes() == final_bytes


def test_scenario_verify_6800_case_clustered_intervals_and_controls(
    sources: dict[str, dict], reconstruction: exp.FrozenReconstruction, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6800-CLUSTERED-INFERENCE keeps cases as bootstrap units."""

    units = [
        exp.project_transfer_unit(group, transformation)
        for group in sources["exp6799"]["probe_groups"][:2]
        for transformation in exp.TRANSFORMATIONS
    ]
    manifest = exp.frozen_manifest(units, seeds=[exp.FROZEN_SEEDS[0]])
    execution = exp.execute_cells(
        units=units,
        reconstruction=reconstruction,
        checkpoint_path=tmp_path / "small-grid.json",
        manifest=manifest,
        seeds=[exp.FROZEN_SEEDS[0]],
    )
    rows = [envelope["payload"] for envelope in execution["rows"]]
    first = exp.aggregate_rows(rows, bootstrap_resamples=100, bootstrap_seed=42)
    second = exp.aggregate_rows(rows, bootstrap_resamples=100, bootstrap_seed=42)
    assert first == second
    assert set(first["clustered_confidence_intervals"]) == set(exp.TRANSFORMATIONS)
    assert (
        first["clustered_confidence_intervals"]["restructuring"]["resampling_unit"] == "source_case"
    )
    assert first["paired_exact_valid_deltas"]["by_transformation"].keys() == set(
        exp.TRANSFORMATIONS
    )
    assert first["destructive_control_results"]["identical_arm"]["paired_exact_valid_delta"] == 0.0
    assert first["destructive_control_results"]["hard_negative"]["exact_invalid_rate"] == 1.0
    assert exp.row_attribution_errors(rows, manifest) == []


def test_scenario_verify_6800_cluster_unit_keeps_source_models_separate() -> None:
    """SCENARIO-VERIFY-6800-CLUSTERED-INFERENCE uses all 97 source cases."""

    paired = [
        {
            "transformation": transformation,
            "case_cluster_key": "shared-formal-case",
            "source_case_id": source_case,
            "exact_valid_delta": delta,
        }
        for transformation in exp.TRANSFORMATIONS
        for source_case, delta in (("model-a|case", 0.0), ("model-b|case", 1.0))
    ]
    intervals = exp._clustered_intervals(paired, resamples=0, seed=1)
    assert all(interval["case_count"] == 2 for interval in intervals.values())
    assert all(interval["point"] == 0.5 for interval in intervals.values())


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the full authentic transfer once for all terminal artifact checks."""

    root = tmp_path_factory.mktemp("exp6800-complete")
    return exp.build_artifact(
        run_date="20260831",
        repo_root=REPO_ROOT,
        checkpoint_path=root / "checkpoint.json",
    )


def test_req_verify_6800_complete_artifact_contains_full_real_output_grid(
    complete_artifact: dict,
) -> None:
    """REQ-VERIFY-6800 completes all cells independent of the measured sign."""

    artifact = complete_artifact
    assert exp.validate_artifact(artifact) == []
    assert artifact["model_output_fixed_point_comparison_completed"] is True
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_is_oracle"] is False
    assert len(artifact["rows"]) == exp.PLANNED_ROW_COUNT
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["parameter_counts_by_arm"] == {arm: 91 for arm in exp.ARMS}
    assert artifact["optimization_steps_by_arm"] == {arm: 30 for arm in exp.ARMS}
    assert artifact["candidate_budget_by_arm"] == {arm: 4365 for arm in exp.ARMS}
    assert artifact["work_matching"]["planned_budgets_match"] is True
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)

    observed = {
        (
            row["source_model"],
            row["constraint_family"],
            row["case_cluster_key"],
            row["transformation"],
            row["random_seed"],
            row["arm"],
        )
        for row in artifact["rows"]
    }
    assert len(observed) == exp.PLANNED_ROW_COUNT
    assert all(
        row["exact_evaluation_receipt"]["evaluated_after_proposal"] for row in artifact["rows"]
    )


def test_req_verify_6800_validation_refuses_drift(complete_artifact: dict) -> None:
    """REQ-VERIFY-6800 rejects schema, row, checksum, completion, and oracle drift."""

    mutations = (
        (lambda value: value.pop("schema"), "required field set mismatch"),
        (lambda value: value["field_principles"].pop("rows"), "field principle coverage"),
        (lambda value: value.__setitem__("inference_substrate", "bad"), "inference substrate"),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s"),
        (lambda value: value.__setitem__("random_seed", -1), "random seed"),
        (lambda value: value.__setitem__("verdict_class", "bad"), "closed enum"),
        (lambda value: value.__setitem__("honest_verdict", "bad"), "terminal prefix"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (lambda value: value.__setitem__("rows", value["rows"][:-1]), "row count"),
        (
            lambda value: value.__setitem__("model_output_fixed_point_comparison_completed", False),
            "completion flag",
        ),
        (lambda value: value.__setitem__("reproducibility_checksum", "bad"), "checksum"),
    )
    for mutate, expected in mutations:
        changed = deepcopy(complete_artifact)
        mutate(changed)
        assert any(expected in error for error in exp.validate_artifact(changed))


def test_scenario_verify_6800_blocked_artifact_and_cli_are_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-6800-BLOCKED writes full schema and no fallback rows."""

    output = tmp_path / "blocked.json"
    checkpoint = tmp_path / "checkpoint.json"
    assert (
        exp.main(
            [
                "--date",
                "20260831",
                "--repo-root",
                str(tmp_path),
                "--artifact-path",
                str(output),
                "--checkpoint-path",
                str(checkpoint),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "complete_blocked_real_output_fixed_point_transfer"
    assert artifact["rows"] == []
    assert artifact["model_output_fixed_point_comparison_completed"] is False
    assert artifact["verdict_class"] == "blocked"
    assert exp.validate_artifact(artifact) == []
    assert "complete_blocked_real_output_fixed_point_transfer" in capsys.readouterr().out

    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(run_date="2026-08-31", repo_root=tmp_path)


def test_req_verify_6800_defensive_inputs_fail_closed(
    sources: dict[str, dict],
    reconstruction: exp.FrozenReconstruction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6800 covers malformed inputs, drift, controls, and attribution."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        exp.load_json_object(scalar)
    with pytest.raises(ValueError, match="unknown transformation"):
        exp.project_transfer_unit(sources["exp6799"]["probe_groups"][0], "unknown")

    unit = exp.project_transfer_unit(sources["exp6799"]["probe_groups"][0], "base")
    with pytest.raises(ValueError, match="unknown control"):
        exp._controlled_unit(unit, "unknown")
    denied = deepcopy(unit)
    denied["proposal_features"]["exact_valid"] = True
    with pytest.raises(ValueError, match="oracle feature refusal"):
        exp.propose_transfer_row(
            reconstruction.models[(exp.FROZEN_SEEDS[0], exp.FLAT_ARM)],
            denied,
            arm=exp.FLAT_ARM,
            seed=exp.FROZEN_SEEDS[0],
        )

    bad_sources = deepcopy(sources)
    bad_sources["exp6788"]["rows"][0]["candidate_hashes"] = ["sha256:drift"]
    with pytest.raises(exp.ReproductionError, match="candidate mismatch"):
        exp.reconstruct_frozen_arms(bad_sources)

    manifest = exp.frozen_manifest([unit], seeds=[exp.FROZEN_SEEDS[0]])
    execution = exp.execute_cells(
        units=[unit],
        reconstruction=reconstruction,
        checkpoint_path=tmp_path / "attribution.json",
        manifest=manifest,
        seeds=[exp.FROZEN_SEEDS[0]],
    )
    rows = [row["payload"] for row in execution["rows"]]
    duplicate = [rows[0], rows[0]]
    assert "duplicate row IDs" in exp.row_attribution_errors(duplicate, manifest)
    changed = deepcopy(rows)
    changed[0]["row_id"] = "wrong"
    changed[0]["candidate_budget"] = 99
    errors = exp.row_attribution_errors(changed, manifest)
    assert any("identity mismatch" in error for error in errors)
    assert any("candidate budget mismatch" in error for error in errors)
    assert "each paired key must contain both arms" in exp.row_attribution_errors(
        rows[:1], manifest
    )
    assert exp._paired_values(rows[:1]) == []

    assert exp.percentile([2.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="nonempty"):
        exp.percentile([], 0.5)
    with pytest.raises(ValueError, match="quantile"):
        exp.percentile([1.0], 2.0)


def test_scenario_verify_6800_build_and_validation_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6800-BLOCKED tests every construction guard."""

    blocked = exp.build_artifact(repo_root=tmp_path)
    malformed_blocked = deepcopy(blocked)
    malformed_blocked["status"] = "bad"
    malformed_blocked["rows"] = [{}]
    malformed_blocked["model_output_fixed_point_comparison_completed"] = True
    malformed_blocked["gate_check_summary"]["all_passed"] = True
    malformed_blocked["reproducibility_checksum"] = exp.reproducibility_checksum(malformed_blocked)
    errors = exp.validate_artifact(malformed_blocked)
    assert "blocked artifact status mismatch" in errors
    assert "blocked artifact must not contain rows" in errors
    assert "blocked artifact completion flag mismatch" in errors
    assert "blocked artifact must name failed gates" in errors

    completed = deepcopy(blocked)
    completed.update(
        {
            "status": "complete",
            "verdict_class": "positive",
            "honest_verdict": "complete: forced validation case",
            "rows": [],
            "model_output_fixed_point_comparison_completed": True,
            "decision_gates": {"positive": False},
        }
    )
    completed["gate_check_summary"]["all_passed"] = False
    completed["reproducibility_checksum"] = exp.reproducibility_checksum(completed)
    with monkeypatch.context() as patch:
        patch.setattr(exp, "row_attribution_errors", lambda _rows, _manifest: [])
        errors = exp.validate_artifact(completed)
    assert "complete artifact has failed preconditions" in errors
    assert "positive verdict does not match decision gates" in errors

    good_summary = {
        "all_passed": True,
        "checks": [],
        "failed_checks": [],
        "first_failure": None,
        "source_artifact_hashes": {},
    }
    with monkeypatch.context() as patch:
        patch.setattr(exp, "evaluate_preconditions", lambda **_kwargs: good_summary)
        patch.setattr(exp, "load_sources", lambda _root: {})
        patch.setattr(
            exp,
            "reconstruct_frozen_arms",
            lambda _sources: (_ for _ in ()).throw(exp.ReproductionError("forced drift")),
        )
        reproduction_block = exp.build_artifact(repo_root=tmp_path)
    assert reproduction_block["verdict_class"] == "blocked"
    assert reproduction_block["gate_check_summary"]["first_failure"]["check"] == (
        "v592_candidate_hash_reproduction"
    )

    empty_reconstruction = exp.FrozenReconstruction({}, [], [])
    with monkeypatch.context() as patch:
        patch.setattr(exp, "evaluate_preconditions", lambda **_kwargs: good_summary)
        patch.setattr(exp, "load_sources", lambda _root: {"exp6799": {}})
        patch.setattr(exp, "reconstruct_frozen_arms", lambda _sources: empty_reconstruction)
        patch.setattr(exp, "build_transfer_units", lambda _source: [])
        patch.setattr(exp, "audit_transfer_feature_contract", lambda _units: ["denied"])
        feature_block = exp.build_artifact(repo_root=tmp_path)
    assert feature_block["verdict_class"] == "blocked"
    assert feature_block["gate_check_summary"]["first_failure"]["check"] == (
        "transfer_feature_contract"
    )

    valid_empty_aggregates = {
        **exp._empty_aggregates(),
        "clustered_confidence_intervals": {
            transformation: {"lower": 0.0} for transformation in exp.TRANSFORMATIONS
        },
        "support_contraction": {},
        "convergence_harm": {},
        "work_matching": {
            "planned_budgets_match": True,
            "no_grouped_work_harm": True,
        },
    }
    with monkeypatch.context() as patch:
        patch.setattr(exp, "evaluate_preconditions", lambda **_kwargs: good_summary)
        patch.setattr(exp, "load_sources", lambda _root: {"exp6799": {}})
        patch.setattr(exp, "reconstruct_frozen_arms", lambda _sources: empty_reconstruction)
        patch.setattr(exp, "build_transfer_units", lambda _source: [])
        patch.setattr(exp, "audit_transfer_feature_contract", lambda _units: [])
        patch.setattr(
            exp,
            "execute_cells",
            lambda **_kwargs: {
                "rows": [],
                "pending_row_ids": [],
                "new_row_count": 0,
                "manifest_hash": "sha256:empty",
            },
        )
        patch.setattr(exp, "aggregate_rows", lambda _rows: valid_empty_aggregates)
        patch.setattr(exp, "parameter_counts", lambda _models: {arm: 91 for arm in exp.ARMS})
        patch.setattr(exp, "validate_artifact", lambda _artifact: ["forced validation error"])
        with pytest.raises(ValueError, match="forced validation error"):
            exp.build_artifact(repo_root=tmp_path, checkpoint_path=tmp_path / "unused.json")


def test_scenario_verify_6800_construction_validation_errors_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6800-BLOCKED refuses invalid artifacts it constructs."""

    with monkeypatch.context() as patch:
        patch.setattr(exp, "validate_artifact", lambda _artifact: ["forced blocked error"])
        with pytest.raises(ValueError, match="forced blocked error"):
            exp.build_artifact(repo_root=tmp_path)

    good_summary = {
        "all_passed": True,
        "checks": [],
        "failed_checks": [],
        "first_failure": None,
        "source_artifact_hashes": {},
    }
    with monkeypatch.context() as patch:
        patch.setattr(exp, "evaluate_preconditions", lambda **_kwargs: good_summary)
        patch.setattr(exp, "load_sources", lambda _root: {})
        patch.setattr(
            exp,
            "reconstruct_frozen_arms",
            lambda _sources: (_ for _ in ()).throw(exp.ReproductionError("forced drift")),
        )
        patch.setattr(exp, "validate_artifact", lambda _artifact: ["forced replay error"])
        with pytest.raises(ValueError, match="forced replay error"):
            exp.build_artifact(repo_root=tmp_path)
