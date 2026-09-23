"""Tests for REQ-CL-7576 and SCENARIO-CL-7576-*.

The fixtures stay private. They test the fit contract without rewriting the
repository's historical result files.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7576_v662_proper_loss_energy as subject
from carnot.experiment_7561_v661_recalibration_prototype import KNOTS, map_probability
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def _row(role: str, index: int, probability: float, label: int) -> dict:
    return {
        "source_id": f"{role}-{index}",
        "group_id": f"group-{role}-{index}",
        "role": role,
        "official_split": "train" if role != "policy" else "validation",
        "probability": probability,
        "label": label,
        "request_hashes": [f"request-{role}-{index}-{cell}" for cell in range(6)],
        "prompt_hashes": [f"prompt-{role}-{index}-{cell}" for cell in range(6)],
        "diagnostics_are_labels": False,
    }


def _roles(count: int = 24) -> dict[str, list[dict]]:
    probabilities = np.linspace(0.02, 0.98, count)
    fit = [
        _row("fit", index, float(value), int(index % 3 == 0 or value > 0.72))
        for index, value in enumerate(probabilities)
    ]
    tune = [
        _row("tune", index, float(value), int(index % 4 == 0 or value > 0.68))
        for index, value in enumerate(probabilities[::2])
    ]
    policy = [
        _row("policy", index, float(value), int(value > 0.5))
        for index, value in enumerate(probabilities[1::2])
    ]
    return {"fit": fit, "tune": tune, "policy": policy}


def _bundle() -> dict:
    roles = _roles()
    return subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])


def _passing_receipts() -> list[dict]:
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"command-{name}",
            "worktree": str(subject.REPO_ROOT),
            "log_sha256": canonical_hash(name),
        }
        for name in (*subject.REQUIRED_VALIDATION_NAMES, *subject.TERMINAL_CHECK_NAMES)
    ]


def _passing_checks() -> list[dict]:
    return [
        {
            "check": "fixture",
            "upstream": "fixture",
            "path": "fixture.json",
            "field": "ready",
            "op": "==",
            "expected": True,
            "observed": True,
            "passed": True,
        }
    ]


def test_energy_normalization_clips_only_log_evaluation() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-ENERGY."""

    for probability in (0.0, 0.2, 0.8, 1.0):
        energies = subject.normalized_binary_energies(probability)
        expected = min(1.0 - subject.LOG_CLIP, max(subject.LOG_CLIP, probability))
        assert math.isclose(subject.probability_from_energies(energies), expected, abs_tol=1e-12)
        assert energies["unclipped_probability"] == probability
    with pytest.raises(ValueError, match="probability_not_finite"):
        subject.normalized_binary_energies(math.nan)
    with pytest.raises(ValueError, match="energy_pair_invalid"):
        subject.probability_from_energies({"correct": 0.0})


def test_candidate_and_range_only_control_fit_registered_objective() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-FIT and CONTROLS."""

    bundle = _bundle()
    candidate = bundle["heads"]["proper_loss_monotone"]
    range_only = bundle["heads"]["unconstrained_nine_knot"]
    assert candidate["solver_receipt"]["converged"] is True
    assert candidate["constraint_residuals"]["maximum"] <= 1e-7
    assert max(abs(a - b) for a, b in zip(candidate["theta"], KNOTS, strict=True)) <= 0.1000001
    assert np.all(np.diff(candidate["theta"]) >= -1e-7)
    assert range_only["solver_receipt"]["converged"] is True
    assert all(0.0 <= value <= 1.0 for value in range_only["theta"])
    assert bundle["fit_source_ids_sha256"] == canonical_hash(
        [row["source_id"] for row in _roles()["fit"]]
    )
    assert bundle["strongest_comparator"]["selected_on_role"] == "tune"
    assert bundle["heads_frozen_before_policy_access"] is True


def test_fit_rows_reject_online_test_and_duplicate_identities() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-ISOLATION."""

    roles = _roles()
    bad = deepcopy(roles["fit"])
    bad[0]["role"] = "online"
    with pytest.raises(ValueError, match="fit_role_invalid"):
        subject.fit_proper_loss_bundle(bad, roles["tune"], roles["policy"])
    duplicate = deepcopy(roles["fit"])
    duplicate[1]["source_id"] = duplicate[0]["source_id"]
    with pytest.raises(ValueError, match="source_id_duplicate"):
        subject.fit_proper_loss_bundle(duplicate, roles["tune"], roles["policy"])
    with pytest.raises(ValueError, match="forbidden_role_requested"):
        subject.load_frozen_fit_roles(Path("."), [], requested_roles=("fit", "test"))


def test_rows_reduce_from_raw_numerators_and_keep_direction() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-CONTROLS."""

    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    rows = subject.build_comparison_rows(bundle, roles["fit"], roles["tune"])
    reduced = subject.reduce_comparison_rows(rows)
    assert len(rows) == (len(roles["fit"]) + len(roles["tune"])) * 4
    assert set(reduced) == {
        "proper_loss_monotone",
        "raw_original",
        "temperature_original",
        "unconstrained_nine_knot",
    }
    assert all(row["raw_squared_error_denominator"] == 1 for row in rows)
    assert all(row["metric_direction"] == "lower_brier_is_better" for row in rows)
    broken = deepcopy(rows)
    broken[0]["raw_squared_error_numerator"] += 0.1
    with pytest.raises(ValueError, match="row_numerator_invalid"):
        subject.reduce_comparison_rows(broken)
    broken = deepcopy(rows)
    broken[0]["raw_squared_error_denominator"] = 2
    with pytest.raises(ValueError, match="row_denominator_invalid"):
        subject.reduce_comparison_rows(broken)


def test_structural_controls_and_noise_are_diagnostic_only() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-DIAGNOSTICS."""

    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    assert controls["passed"] is True
    assert controls["identity_reproduction"]["maximum_error"] <= 1e-12
    assert controls["option_order_mapping"]["six_cell_rows"] == len(roles["fit"])
    assert controls["parameter_sensitivity"]["all_knots_responsive"] is True
    assert controls["shuffled_labels"]["empirical_benefit_claim"] is False
    assert [row["perturbation"] for row in controls["input_sensitivity"]] == [0.0, 0.01, 0.05]
    assert controls["input_sensitivity"][0]["maximum_absolute_change"] == 0.0
    assert all(row["diagnostic_only"] is True for row in controls["input_sensitivity"])


def test_policy_access_cannot_select_or_refit_heads() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-CONTROLS."""

    roles = _roles()
    first = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    changed_policy = deepcopy(roles["policy"])
    for row in changed_policy:
        row["label"] = 1 - row["label"]
    second = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], changed_policy)
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert first["strongest_comparator"] == second["strongest_comparator"]
    assert first["policy_action_contract"] == second["policy_action_contract"]
    assert first["policy_action_contract"]["labels_consumed"] == 0


def test_solver_failure_uses_named_identity_control_and_closes_readiness() -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-FIT."""

    roles = _roles()

    def failed_solver(_gram: np.ndarray, _target: np.ndarray) -> tuple[np.ndarray, dict]:
        return KNOTS.copy(), {
            "method": "registered_failure_fixture",
            "converged": False,
            "objective": 0.0,
            "constraint_errors": ["fixture_failure"],
        }

    bundle = subject.fit_proper_loss_bundle(
        roles["fit"], roles["tune"], roles["policy"], candidate_solver=failed_solver
    )
    assert bundle["candidate_fallback"]["used"] is True
    assert bundle["candidate_fallback"]["name"] == "identity_forecast_control"
    assert bundle["proper_loss_fit_ready_score"] == 0
    assert bundle["trained_benefit_claimed"] is False
    assert bundle["heads"]["proper_loss_monotone"]["theta"] == KNOTS.tolist()


def test_checkpoint_persist_reload_lifecycle(tmp_path: Path) -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-TERMINAL."""

    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    receipts = subject.write_fit_sidecars(tmp_path, bundle, controls, root=tmp_path)
    reloaded = subject.reload_checkpoint(tmp_path / subject.CHECKPOINT_PATH)
    assert receipts["checkpoint"]["sha256"] == sha256_file(tmp_path / subject.CHECKPOINT_PATH)
    assert reloaded["checkpoint_sha256"] == bundle["checkpoint_sha256"]
    assert reloaded["heads"] == bundle["heads"]
    assert reloaded["lifecycle"] == ["predict", "release", "update", "persist", "reload"]


def test_frozen_exp7575_reader_authenticates_and_filters_roles(tmp_path: Path) -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-ISOLATION."""

    roles = _roles(8)
    all_rows = [*roles["fit"], *roles["tune"], *roles["policy"]]
    all_rows.extend([_row("online", 0, 0.4, 0), _row("test", 0, 0.6, 1)])
    sidecar = tmp_path / "roles.jsonl"
    sidecar.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows), encoding="utf-8"
    )
    artifact_path = tmp_path / subject.UPSTREAM_PROTOCOL_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        artifact_path,
        {
            "raw_sidecars": {
                "cached_roles": {
                    "path": "roles.jsonl",
                    "sha256": sha256_file(sidecar),
                    "rows": len(all_rows),
                }
            }
        },
    )
    hashes: list[dict] = []
    loaded = subject.load_frozen_fit_roles(
        tmp_path,
        hashes,
        expected_counts={name: len(rows) for name, rows in roles.items()},
    )
    assert set(loaded) == {"fit", "tune", "policy"}
    assert sum(map(len, loaded.values())) == sum(map(len, roles.values()))
    assert all(row["role"] not in {"online", "test"} for rows in loaded.values() for row in rows)
    assert {Path(row["path"]).name for row in hashes} == {
        subject.UPSTREAM_PROTOCOL_PATH.name,
        sidecar.name,
    }
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cached_roles_hash_invalid"):
        subject.load_frozen_fit_roles(
            tmp_path,
            [],
            expected_counts={name: len(rows) for name, rows in roles.items()},
        )


def test_blocked_artifact_names_exact_missing_operand(tmp_path: Path) -> None:
    """REQ-CL-7576 blocked external evidence contract."""

    checks, hashes = subject.collect_preconditions(tmp_path)
    artifact = subject.build_blocked_artifact(checks, hashes, duration_s=0.1)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["proper_loss_fit_ready_score"] == 0
    first = artifact["gate_check_summary"]["first_failure"]
    assert {"check", "upstream", "path", "field", "op", "expected", "observed"} <= set(first)
    assert subject.validate_artifact(artifact, root=tmp_path, require_validation=False)["blocked"]


def test_artifact_readiness_reduces_without_claiming_benefit(tmp_path: Path) -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-TERMINAL."""

    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    rows = subject.build_comparison_rows(bundle, roles["fit"], roles["tune"])
    sidecars = subject.write_fit_sidecars(tmp_path, bundle, controls, root=tmp_path)
    artifact = subject.build_artifact(
        bundle=bundle,
        controls=controls,
        rows=rows,
        preconditions=_passing_checks(),
        source_hashes=[],
        sidecars=sidecars,
        validation_receipts=_passing_receipts(),
        duration_s=1.25,
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, artifact)
    validated = subject.cold_replay(path, root=tmp_path)
    reduced = subject.independent_reduce_artifact(path, root=tmp_path)
    assert validated["valid"] is True
    assert reduced["proper_loss_fit_ready_score"] == 1
    assert reduced["baseline_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_proper_loss_heads_frozen_benefit_unmeasured"
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["fit_brier"] == bundle["heads"]["proper_loss_monotone"]["fit_brier"]
    assert artifact["head_manifest_path"] == sidecars["head_manifest"]["path"]
    assert artifact["MODEL_SPECS"] == [] and artifact["no_model_load"] is True
    assert artifact["invocation_counts"] == subject.ZERO_INVOCATION_COUNTS


def test_artifact_mutations_fail_cold_validation(tmp_path: Path) -> None:
    """REQ-CL-7576 / SCENARIO-CL-7576-TERMINAL."""

    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    rows = subject.build_comparison_rows(bundle, roles["fit"], roles["tune"])
    sidecars = subject.write_fit_sidecars(tmp_path, bundle, controls, root=tmp_path)
    artifact = subject.build_artifact(
        bundle=bundle,
        controls=controls,
        rows=rows,
        preconditions=_passing_checks(),
        source_hashes=[],
        sidecars=sidecars,
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
    )
    for mutation, error in (
        (("MODEL_SPECS", ["historical-model"]), "model_specs_not_empty"),
        (("fresh_confirmatory_claim_allowed", True), "freshness_invalid"),
        (("fit_brier", artifact["fit_brier"] + 0.1), "fit_brier_mismatch"),
    ):
        broken = deepcopy(artifact)
        broken[mutation[0]] = mutation[1]
        broken["reproducibility_checksum"] = subject.artifact_checksum(broken)
        with pytest.raises(ValueError, match=error):
            subject.validate_artifact(broken, root=tmp_path)


def test_fail_closed_input_and_role_guards(tmp_path: Path) -> None:
    """REQ-CL-7576 rejects malformed probabilities, labels, and custody."""

    with pytest.raises(ValueError, match="probability_out_of_range"):
        subject.normalized_binary_energies(1.1)
    with pytest.raises(ValueError, match="binary_label_required"):
        subject._brier(0.5, 2)
    invalid_json = tmp_path / "array.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        subject._load_object(invalid_json)
    outside = tmp_path.parent / "outside-exp7576.txt"
    outside.write_text("outside", encoding="utf-8")
    assert Path(subject._source_hash(outside, tmp_path)["path"]).is_absolute()

    role_rows = _roles()["fit"]
    with pytest.raises(ValueError, match="forbidden_role_requested"):
        subject._validate_role_rows(role_rows, "test")
    with pytest.raises(ValueError, match="fit_rows_empty"):
        subject._validate_role_rows([], "fit")
    for field, value, error in (
        ("label", 2, "binary_label_required"),
        ("request_hashes", ["same"] * 6, "option_order_custody_invalid"),
        ("diagnostics_are_labels", True, "diagnostic_label_leakage"),
    ):
        broken = deepcopy(role_rows)
        broken[0][field] = value
        with pytest.raises(ValueError, match=error):
            subject._validate_role_rows(broken, "fit")
    unsupported = deepcopy(role_rows)
    for row in unsupported:
        row["label"] = 0
    with pytest.raises(ValueError, match="fit_binary_support_invalid"):
        subject._validate_role_rows(unsupported, "fit")


def test_frozen_reader_rejects_receipt_rows_and_counts(tmp_path: Path) -> None:
    """REQ-CL-7576 authenticates the complete Exp7575 row envelope."""

    artifact = tmp_path / subject.UPSTREAM_PROTOCOL_PATH
    artifact.parent.mkdir(parents=True)
    atomic_json(artifact, {"raw_sidecars": {"cached_roles": "missing"}})
    with pytest.raises(ValueError, match="cached_roles_receipt_missing"):
        subject.load_frozen_fit_roles(tmp_path, [])

    sidecar = tmp_path / "bad.jsonl"
    sidecar.write_text("[]\n", encoding="utf-8")
    atomic_json(
        artifact,
        {
            "raw_sidecars": {
                "cached_roles": {"path": "bad.jsonl", "sha256": sha256_file(sidecar), "rows": 1}
            }
        },
    )
    with pytest.raises(ValueError, match="cached_role_row_invalid"):
        subject.load_frozen_fit_roles(
            tmp_path, [], expected_counts={name: 0 for name in subject.ALLOWED_ROLES}
        )

    rows = _roles(8)
    payload = [row for values in rows.values() for row in values]
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    receipt = {"path": "bad.jsonl", "sha256": sha256_file(sidecar), "rows": len(payload) + 1}
    atomic_json(artifact, {"raw_sidecars": {"cached_roles": receipt}})
    with pytest.raises(ValueError, match="cached_roles_count_invalid"):
        subject.load_frozen_fit_roles(
            tmp_path, [], expected_counts={name: len(values) for name, values in rows.items()}
        )
    receipt["rows"] = len(payload)
    atomic_json(artifact, {"raw_sidecars": {"cached_roles": receipt}})
    with pytest.raises(ValueError, match="role_count_invalid:fit"):
        subject.load_frozen_fit_roles(
            tmp_path, [], expected_counts={"fit": 99, "tune": 4, "policy": 4}
        )


def _artifact_fixture(tmp_path: Path) -> dict:
    roles = _roles()
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    sidecars = subject.write_fit_sidecars(tmp_path, bundle, controls, root=tmp_path)
    return subject.build_artifact(
        bundle=bundle,
        controls=controls,
        rows=subject.build_comparison_rows(bundle, roles["fit"], roles["tune"]),
        preconditions=_passing_checks(),
        source_hashes=[],
        sidecars=sidecars,
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
    )


def test_solver_exception_and_comparison_reduction_guards() -> None:
    """REQ-CL-7576 retains identity only as a failed-solver control."""

    roles = _roles()

    def exploding_solver(_gram: np.ndarray, _target: np.ndarray) -> tuple[np.ndarray, dict]:
        raise RuntimeError("fixture")

    bundle = subject.fit_proper_loss_bundle(
        roles["fit"], roles["tune"], roles["policy"], candidate_solver=exploding_solver
    )
    assert bundle["candidate_fallback"]["used"] is True
    assert (
        bundle["heads"]["proper_loss_monotone"]["solver_receipt"]["method"]
        == "registered_solver_exception"
    )
    with pytest.raises(ValueError, match="comparison_arm_invalid"):
        subject._arm_probability(bundle, "unknown", 0.5)
    with pytest.raises(ValueError, match="comparison_rows_empty"):
        subject.reduce_comparison_rows([])
    rows = subject.build_comparison_rows(bundle, roles["fit"], roles["tune"])
    rows[0]["metric_direction"] = "higher"
    with pytest.raises(ValueError, match="row_direction_invalid"):
        subject.reduce_comparison_rows(rows)


def test_checkpoint_and_sidecar_mutations_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7576 binds checkpoint, control, and manifest bytes."""

    artifact = _artifact_fixture(tmp_path)
    checkpoint_path = tmp_path / subject.CHECKPOINT_PATH
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    broken = deepcopy(checkpoint)
    broken["checkpoint_payload"]["schema"] = "changed"
    atomic_json(checkpoint_path, broken)
    with pytest.raises(ValueError, match="checkpoint_hash_invalid"):
        subject.reload_checkpoint(checkpoint_path)
    broken = deepcopy(checkpoint)
    broken["heads"] = {}
    atomic_json(checkpoint_path, broken)
    with pytest.raises(ValueError, match="checkpoint_heads_invalid"):
        subject.reload_checkpoint(checkpoint_path)
    broken = deepcopy(checkpoint)
    broken["lifecycle"] = []
    atomic_json(checkpoint_path, broken)
    with pytest.raises(ValueError, match="checkpoint_lifecycle_invalid"):
        subject.reload_checkpoint(checkpoint_path)
    atomic_json(checkpoint_path, checkpoint)

    broken_artifact = deepcopy(artifact)
    broken_artifact["raw_sidecars"]["checkpoint"]["sha256"] = "sha256:bad"
    broken_artifact["reproducibility_checksum"] = subject.artifact_checksum(broken_artifact)
    with pytest.raises(ValueError, match="sidecar_hash_invalid:checkpoint"):
        subject.validate_artifact(broken_artifact, root=tmp_path)
    broken_artifact = deepcopy(artifact)
    broken_artifact["fit_bundle"]["checkpoint_sha256"] = "sha256:bad"
    broken_artifact["reproducibility_checksum"] = subject.artifact_checksum(broken_artifact)
    with pytest.raises(ValueError, match="checkpoint_identity_invalid"):
        subject.validate_artifact(broken_artifact, root=tmp_path)


def test_terminal_validator_rejects_claim_and_receipt_mutations(tmp_path: Path) -> None:
    """REQ-CL-7576 independently rejects invalid terminal declarations."""

    artifact = _artifact_fixture(tmp_path)
    mutations = (
        ("schema", "wrong", "artifact_identity_invalid"),
        ("no_model_load", False, "no_model_load_invalid"),
        ("invocation_counts", {}, "invocation_counts_invalid"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "null", "terminal_prefix_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("comparative_reduction", {}, "comparative_reduction_mismatch"),
        ("proper_loss_fit_ready_score", 0, "proper_loss_fit_readiness_mismatch"),
        ("baseline_ready_score", 0, "baseline_readiness_mismatch"),
        ("verdict_class", "disqualified", "verdict_reduction_mismatch"),
    )
    for field, value, error in mutations:
        broken = deepcopy(artifact)
        broken[field] = value
        broken["reproducibility_checksum"] = subject.artifact_checksum(broken)
        with pytest.raises(ValueError, match=error):
            subject.validate_artifact(broken, root=tmp_path)
    checksum_broken = deepcopy(artifact)
    checksum_broken["fit_brier"] += 0.01
    with pytest.raises(ValueError, match="artifact_checksum_invalid"):
        subject.validate_artifact(checksum_broken, root=tmp_path)

    with pytest.raises(ValueError, match="validation_receipt_missing"):
        subject._validate_receipts([])
    failed = _passing_receipts()
    failed[0]["passed"] = False
    with pytest.raises(ValueError, match="validation_receipt_failed"):
        subject._validate_receipts(failed)
    incomplete = _passing_receipts()
    incomplete[0]["command"] = ""
    with pytest.raises(ValueError, match="validation_receipt_incomplete"):
        subject._validate_receipts(incomplete)


def test_registered_preconditions_and_command_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7576 freezes affected files and the exact bounded readers."""

    checks, hashes = subject.collect_preconditions(subject.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes
    for path in (
        *subject.AFFECTED_MANIFEST.test_paths,
        *subject.AFFECTED_MANIFEST.changed_modules,
        *subject.AFFECTED_MANIFEST.static_paths,
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path, encoding="utf-8")
    manifest = subject._write_affected_manifest(tmp_path)
    assert manifest["worktree"] == str(tmp_path)
    assert set(manifest["file_hashes"]) == {
        *subject.AFFECTED_MANIFEST.test_paths,
        *subject.AFFECTED_MANIFEST.changed_modules,
        *subject.AFFECTED_MANIFEST.static_paths,
    }

    class PrivateCommand:
        argv = (
            "pytest",
            f"--basetemp={tmp_path / 'pytest' / 'case'}",
            f"--data-file={tmp_path / 'coverage' / '.coverage'}",
        )
        command_environment = (("COVERAGE_FILE", str(tmp_path / "report" / ".coverage")),)

    subject._prepare_private_parents(PrivateCommand())  # type: ignore[arg-type]
    assert (tmp_path / "pytest").is_dir()
    assert (tmp_path / "coverage").is_dir()
    assert (tmp_path / "report").is_dir()
    terminal = subject._terminal_commands(subject.TERMINAL_CANDIDATE_PATH)
    assert [row.spec.name for row in terminal] == list(subject.TERMINAL_CHECK_NAMES)
    subject.progress(0.0, "fixture", "boundary", units=1)
    assert "phase=fixture event=boundary" in capsys.readouterr().out


def test_disqualified_artifact_and_blocked_validation_guards(tmp_path: Path) -> None:
    """REQ-CL-7576 closes readiness after solver or blocked-row failure."""

    roles = _roles()

    def failed_solver(_gram: np.ndarray, _target: np.ndarray) -> tuple[np.ndarray, dict]:
        return KNOTS.copy(), {"method": "fixture", "converged": False}

    bundle = subject.fit_proper_loss_bundle(
        roles["fit"], roles["tune"], roles["policy"], candidate_solver=failed_solver
    )
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    sidecars = subject.write_fit_sidecars(tmp_path, bundle, controls, root=tmp_path)
    artifact = subject.build_artifact(
        bundle=bundle,
        controls=controls,
        rows=subject.build_comparison_rows(bundle, roles["fit"], roles["tune"]),
        preconditions=_passing_checks(),
        source_hashes=[],
        sidecars=sidecars,
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert subject.validate_artifact(artifact, root=tmp_path)["valid"] is True

    blocked = subject.build_blocked_artifact(_passing_checks(), [], duration_s=0.1)
    blocked["rows"] = [{"unexpected": True}]
    blocked["reproducibility_checksum"] = subject.artifact_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_measurement_invalid"):
        subject.validate_artifact(blocked, root=tmp_path, require_validation=False)
    blocked = subject.build_blocked_artifact([], [], duration_s=0.1)
    blocked["gate_check_summary"]["first_failure"] = {}
    blocked["reproducibility_checksum"] = subject.artifact_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_gate_summary_invalid"):
        subject.validate_artifact(blocked, root=tmp_path, require_validation=False)


def test_shuffle_fallback_and_internal_sidecar_hash_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7576 keeps shuffled labels distinct and authenticates inner hashes."""

    roles = _roles()

    class IdentityPermutation:
        @staticmethod
        def permutation(values: list[int]) -> np.ndarray:
            return np.asarray(values)

    monkeypatch.setattr(subject.np.random, "default_rng", lambda _seed: IdentityPermutation())
    bundle = subject.fit_proper_loss_bundle(roles["fit"], roles["tune"], roles["policy"])
    assert bundle["shuffled_label_control"]["label_permutation_sha256"] == canonical_hash(
        [int(row["label"]) for row in roles["fit"]][1:] + [int(roles["fit"][0]["label"])]
    )
    controls = subject.run_qualification_controls(bundle, roles["fit"])
    bad_bundle = deepcopy(bundle)
    bad_bundle["checkpoint_sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="checkpoint_hash_invalid"):
        subject.write_fit_sidecars(tmp_path, bad_bundle, controls, root=tmp_path)

    artifact = _artifact_fixture(tmp_path)
    controls_path = tmp_path / subject.CONTROL_PATH
    persisted_controls = json.loads(controls_path.read_text(encoding="utf-8"))
    persisted_controls["passed"] = not persisted_controls["passed"]
    atomic_json(controls_path, persisted_controls)
    artifact["raw_sidecars"]["qualification_controls"]["sha256"] = sha256_file(controls_path)
    artifact["reproducibility_checksum"] = subject.artifact_checksum(artifact)
    with pytest.raises(ValueError, match="controls_hash_invalid"):
        subject.validate_artifact(artifact, root=tmp_path)

    artifact = _artifact_fixture(tmp_path)
    manifest_path = tmp_path / subject.HEAD_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_before_policy_access"] = False
    atomic_json(manifest_path, manifest)
    artifact["raw_sidecars"]["head_manifest"]["sha256"] = sha256_file(manifest_path)
    artifact["reproducibility_checksum"] = subject.artifact_checksum(artifact)
    with pytest.raises(ValueError, match="head_manifest_hash_invalid"):
        subject.validate_artifact(artifact, root=tmp_path)

    artifact = _artifact_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = "sha256:wrong"
    manifest["manifest_sha256"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    atomic_json(manifest_path, manifest)
    artifact["raw_sidecars"]["head_manifest"]["sha256"] = sha256_file(manifest_path)
    artifact["reproducibility_checksum"] = subject.artifact_checksum(artifact)
    with pytest.raises(ValueError, match="head_manifest_checkpoint_invalid"):
        subject.validate_artifact(artifact, root=tmp_path)
