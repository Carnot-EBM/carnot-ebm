"""Tests for the frozen finite-law empirical sampler audit.

Spec refs: REQ-SAMPLER-7378 and SCENARIO-SAMPLER-7378-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import builtins
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7378_v647_ising_audit as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, object]]:
    """Return one passing receipt for every fixed validation command."""

    return [
        {
            "name": name,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "command_argv": [name],
            "command_environment": {},
            "scope": "unit_fixture",
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
        }
        for name in (*exp.REQUIRED_CHECK_NAMES, *exp.E2E_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def _fixture() -> dict[str, object]:
    upstream = json.loads((ROOT / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))
    return upstream["frozen_formulas"][0]


def test_independent_energy_and_conditioned_reduction_preserve_each_state() -> None:
    """SCENARIO-SAMPLER-7378-SOURCE: reduced sampler inputs keep source energy."""

    formula = {
        "n_vars": 4,
        "original_clauses": [[1, -2], [-1, 3], [-3, 4], [2, 4]],
        "assumptions": [1, -4],
    }
    clauses = formula["original_clauses"]
    parameters = exp.conditioned_sampler_parameters(formula, clauses)

    assert parameters["empty_support"] is False
    assert parameters["free_indices"] == [1, 2]
    assert parameters["fixed_bits"] == {0: 1, 3: 0}
    assert np.asarray(parameters["biases"]).shape == (2,)
    assert np.asarray(parameters["coupling_matrix"]).shape == (2, 2)

    for free_bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
        full = exp.restore_bits(free_bits, parameters)
        direct = exp.independent_source_energy(full, clauses)
        assert exp.reduced_boolean_energy(free_bits, parameters) == pytest.approx(direct, abs=1e-12)

    assert exp.independent_source_energy((1, 0), ((1, -2),)) == 0.0
    assert exp.independent_source_energy((0, 1), ((1, -2),)) == 1.0
    with pytest.raises(ValueError, match="literal_out_of_range"):
        exp.independent_source_energy((1,), ((1, 2),))
    with pytest.raises(ValueError, match="assumption_out_of_range"):
        exp.conditioned_sampler_parameters(
            {"n_vars": 1, "original_clauses": [[1, 1]], "assumptions": [2]},
            [[1, 1]],
        )
    with pytest.raises(ValueError, match="free_state_length"):
        exp.restore_bits((1,), parameters)


def test_empty_conditioned_support_is_explicit_and_never_sampled() -> None:
    """SCENARIO-SAMPLER-7378-EMPTY: contradictions create structural rows."""

    formula = {"n_vars": 2, "original_clauses": [[1, 2]], "assumptions": [1, -1]}
    parameters = exp.conditioned_sampler_parameters(formula, formula["original_clauses"])
    target = exp.enumerated_target(formula, 1.0)

    assert parameters["empty_support"] is True
    assert target["support_size"] == 0
    assert target["normalizer"] == 0.0
    assert target["observables"] == []

    row, trace = exp.run_sampler_chain(
        formula,
        beta=1.0,
        condition="source_only",
        chain_order=0,
        seed=exp.CHAIN_SEEDS[0],
    )
    assert row["outcome"] == "structural_empty_support"
    assert row["recorded_samples"] == 0
    assert row["failures"] == ["empty_conditioned_support_has_no_probability_law"]
    assert trace is None
    empty_cell = exp.aggregate_cell("empty", 1.0, "source_only", [row] * 4, target)
    assert empty_cell["support_status"] == "empty_conditioned_support"
    assert empty_cell["qualified"] is False


def test_shipped_sampler_settings_and_unfiltered_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-7378-PROTOCOL: the existing sampler gets frozen settings."""

    seen: dict[str, object] = {}

    class FakeSampler:
        def __init__(self, **kwargs: object) -> None:
            seen["settings"] = kwargs

        def sample(
            self,
            key: object,
            biases: object,
            coupling_matrix: object,
            beta: float,
        ) -> np.ndarray:
            del key, biases, coupling_matrix
            seen["beta"] = beta
            return np.asarray([[False, False], [True, False], [True, True]], dtype=bool)

    monkeypatch.setattr(exp, "ParallelIsingSampler", FakeSampler)
    formula = {
        "formula_id": "unit",
        "source_hash": "sha256:" + "1" * 64,
        "n_vars": 3,
        "original_clauses": [[1, 2], [-2, 3]],
        "assumptions": [1],
    }
    target = exp.enumerated_target(formula, 0.5)
    row, trace = exp.run_sampler_chain(
        formula,
        beta=0.5,
        condition="proof_assisted_source_only",
        chain_order=2,
        seed=17,
        n_warmup=7,
        n_samples=3,
        steps_per_sample=1,
    )

    assert seen["settings"] == {
        "n_warmup": 7,
        "n_samples": 3,
        "steps_per_sample": 1,
        "schedule": None,
        "use_checkerboard": True,
    }
    assert seen["beta"] == 0.5
    assert trace is not None
    assert trace["state_indices"] == [4, 6, 7]
    assert row["state_counts"] == {"4": 1, "6": 1, "7": 1}
    assert row["support_violation_count"] == 0
    assert row["source_hash"] == formula["source_hash"]
    assert row["chain_order"] == 2
    assert row["recorded_samples"] == 3
    assert row["observable_results"]
    assert row["target_support_size"] == target["support_size"]

    fixed_formula = {
        "formula_id": "fixed",
        "source_hash": "sha256:" + "2" * 64,
        "n_vars": 2,
        "original_clauses": [[1, 2]],
        "assumptions": [1, -2],
    }
    fixed_row, fixed_trace = exp.run_sampler_chain(
        fixed_formula,
        beta=1.0,
        condition="source_only",
        chain_order=0,
        seed=19,
        n_warmup=2,
        n_samples=3,
    )
    assert fixed_row["state_counts"] == {"2": 3}
    assert fixed_trace is not None
    assert fixed_trace["state_indices"] == [2, 2, 2]


def test_autocorrelation_and_cell_observable_gates_stay_separate() -> None:
    """REQ-SAMPLER-7378-QUALIFICATION: error and ESS are independent gates."""

    alternating = [float(index % 2) for index in range(4000)]
    diagnostics = exp.autocorrelation_diagnostics(alternating, target_variance=0.25)
    assert diagnostics["effective_sample_size"] == pytest.approx(4000.0)
    assert diagnostics["lag_correlations"][0] == 1.0
    assert diagnostics["qualified"] is True

    constant = exp.autocorrelation_diagnostics([1.0] * 40, target_variance=0.0)
    assert constant["structurally_degenerate"] is True
    assert constant["effective_sample_size"] is None
    assert constant["qualified"] is True

    stuck = exp.autocorrelation_diagnostics([1.0] * 40, target_variance=0.25)
    assert stuck["constant_observed"] is True
    assert stuck["qualified"] is False
    positively_correlated = exp.autocorrelation_diagnostics(
        [0.0] * 100 + [1.0] * 100 + [0.0] * 100 + [1.0] * 100,
        target_variance=0.25,
    )
    assert positively_correlated["integrated_autocorrelation_time"] > 1.0
    assert exp._support_violation_count(np.zeros((3, 2)), (1, -1)) == 3

    target = {
        "state_probabilities": {"0": 0.5, "1": 0.5},
        "observables": [
            {"observable": "energy_per_clause", "mean": 0.5, "variance": 0.25},
            {"observable": "bit_0", "mean": 0.5, "variance": 0.25},
        ],
    }
    chains = []
    for order in range(4):
        values = [float(index % 2) for index in range(4000)]
        chains.append(
            {
                "chain_order": order,
                "recorded_samples": 4000,
                "outcome": "complete",
                "state_counts": {"0": 2000, "1": 2000},
                "observable_series": {
                    "energy_per_clause": values,
                    "bit_0": values,
                },
                "observable_results": [],
                "failures": [],
            }
        )
    cell = exp.aggregate_cell("unit", 1.0, "source_only", chains, target)
    assert cell["max_observable_error"] == 0.0
    assert cell["observable_error_gate_passed"] is True
    assert cell["effective_sample_gate_passed"] is True
    assert cell["qualified"] is True
    assert cell["interval_coverage"]["covered"] == 2
    assert cell["descriptive_sparse_histogram_tv"] == 0.0

    chains[0]["observable_series"]["bit_0"] = [1.0] * 4000
    biased = exp.aggregate_cell("unit", 1.0, "source_only", chains, target)
    assert biased["max_observable_error"] > 0.05
    assert biased["observable_error_gate_passed"] is False

    degenerate_target = {
        "state_probabilities": {"0": 1.0},
        "observables": [{"observable": "bit_0", "mean": 0.0, "variance": 0.0}],
    }
    degenerate_chains = [
        {
            "chain_order": order,
            "recorded_samples": 4000,
            "outcome": "complete",
            "state_counts": {"0": 4000},
            "observable_series": {"bit_0": [0.0] * 4000},
        }
        for order in range(4)
    ]
    degenerate = exp.aggregate_cell(
        "degenerate", 1.0, "source_only", degenerate_chains, degenerate_target
    )
    assert degenerate["observable_results"][0]["ci95_low"] == 0.0
    assert degenerate["effective_sample_gate_passed"] is True

    unresolved_target = {
        "state_probabilities": {"0": 0.5, "1": 0.5},
        "observables": [{"observable": "bit_0", "mean": 0.5, "variance": 0.25}],
    }
    unresolved = exp.aggregate_cell(
        "unresolved", 1.0, "source_only", degenerate_chains, unresolved_target
    )
    assert unresolved["observable_results"][0]["ci95_low"] is None
    assert unresolved["effective_sample_gate_passed"] is False


def test_appended_clause_control_has_nonzero_exact_distribution_shift() -> None:
    """SCENARIO-SAMPLER-7378-HOSTILE: an entailed extra term changes the law."""

    formula = _fixture()
    source = exp.enumerated_target(formula, 1.0)
    appended_clauses = [*formula["original_clauses"], formula["implied_clause"]]
    appended = exp.enumerated_target(formula, 1.0, clauses=appended_clauses)
    control = exp.distribution_shift_control(formula, 1.0, source, appended)

    assert control["formula_id"] == formula["formula_id"]
    assert control["exact_total_variation"] > 0.0
    assert control["source_satisfying_minima_preserved"] is True
    assert control["wrong_law_detected"] is True
    assert control["appended_clause"] == formula["implied_clause"]


def test_lossless_trace_archive_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    """REQ-SAMPLER-7378-EVIDENCE: ordered raw traces reload independently."""

    records = [
        {
            "cell_id": "a",
            "chain_order": 0,
            "seed": 11,
            "state_indices": [0, 1, 1, 0],
            "trace_sha256": exp.canonical_hash([0, 1, 1, 0]),
        },
        {
            "cell_id": "a",
            "chain_order": 1,
            "seed": 12,
            "state_indices": [1, 1, 0, 0],
            "trace_sha256": exp.canonical_hash([1, 1, 0, 0]),
        },
    ]
    path = tmp_path / "traces.jsonl.gz"
    manifest = exp.atomic_gzip_jsonl(path, records)

    assert manifest["record_count"] == 2
    assert manifest["byte_count"] == path.stat().st_size
    assert manifest["sha256"].startswith("sha256:")
    assert exp.load_trace_archive(path, manifest) == records

    artifact = {
        "raw_trace_archive": manifest,
        "per_chain_results": [
            {
                "cell_id": row["cell_id"],
                "chain_order": row["chain_order"],
                "recorded_samples": len(row["state_indices"]),
                "state_counts": {
                    str(key): value for key, value in Counter(row["state_indices"]).items()
                },
                "trace_sha256": row["trace_sha256"],
            }
            for row in records
        ],
    }
    assert exp.validate_raw_evidence(artifact, tmp_path)["passed"] is True

    relative_manifest = deepcopy(manifest)
    relative_manifest["path"] = path.name
    relative_artifact = {**artifact, "raw_trace_archive": relative_manifest}
    assert exp.validate_raw_evidence(relative_artifact, tmp_path)["passed"] is True

    changed = deepcopy(manifest)
    changed["record_count"] = 3
    with pytest.raises(ValueError, match="trace_archive_record_count"):
        exp.load_trace_archive(path, changed)

    with pytest.raises(ValueError, match="trace_archive_missing"):
        exp.load_trace_archive(tmp_path / "absent.gz", manifest)
    changed = deepcopy(manifest)
    changed["byte_count"] += 1
    with pytest.raises(ValueError, match="trace_archive_byte_count"):
        exp.load_trace_archive(path, changed)
    changed = deepcopy(manifest)
    changed["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="trace_archive_sha256"):
        exp.load_trace_archive(path, changed)

    nonobject_path = tmp_path / "nonobject.jsonl.gz"
    nonobject_manifest = exp.atomic_gzip_jsonl(nonobject_path, [[]])
    with pytest.raises(ValueError, match="trace_archive_row_not_object"):
        exp.load_trace_archive(nonobject_path, nonobject_manifest)

    broken = deepcopy(artifact)
    broken["per_chain_results"][0]["recorded_samples"] = 5
    broken["per_chain_results"][0]["state_counts"] = {"9": 5}
    broken["per_chain_results"][0]["trace_sha256"] = "sha256:wrong"
    reduced = exp.validate_raw_evidence(broken, tmp_path)
    assert reduced["passed"] is False
    assert any(error.startswith("trace_length") for error in reduced["errors"])
    assert any(error.startswith("trace_counts") for error in reduced["errors"])
    assert any(error.startswith("public_trace_hash") for error in reduced["errors"])

    duplicate_path = tmp_path / "duplicate.jsonl.gz"
    duplicate_manifest = exp.atomic_gzip_jsonl(duplicate_path, [records[0], records[0]])
    duplicate = {
        "raw_trace_archive": duplicate_manifest,
        "per_chain_results": artifact["per_chain_results"][:1],
    }
    duplicate_reduction = exp.validate_raw_evidence(duplicate, tmp_path)
    assert any(error.startswith("duplicate_trace") for error in duplicate_reduction["errors"])

    missing_row = {
        "raw_trace_archive": manifest,
        "per_chain_results": artifact["per_chain_results"][:1],
    }
    assert "trace_roster" in exp.validate_raw_evidence(missing_row, tmp_path)["errors"]

    wrong_trace_path = tmp_path / "wrong-trace.jsonl.gz"
    wrong_trace = deepcopy(records[0])
    wrong_trace["trace_sha256"] = "sha256:wrong"
    wrong_trace_manifest = exp.atomic_gzip_jsonl(wrong_trace_path, [wrong_trace])
    wrong_reduction = exp.validate_raw_evidence(
        {
            "raw_trace_archive": wrong_trace_manifest,
            "per_chain_results": artifact["per_chain_results"][:1],
        },
        tmp_path,
    )
    assert any(error.startswith("trace_hash") for error in wrong_reduction["errors"])
    assert exp.validate_raw_evidence({}, tmp_path)["passed"] is False


def test_preconditions_authenticate_upstream_and_fail_closed(tmp_path: Path) -> None:
    """REQ-SAMPLER-7378-PREFLIGHT: exact upstream fields gate dependent work."""

    checks, hashes, upstream, sidecars = exp.collect_preconditions(ROOT)
    blocking = [row for row in checks if row["terminal_blocking"]]
    assert blocking and all(row["passed"] for row in blocking)
    assert upstream["experiment_id"] == "exp7377-v647-ising-law"
    assert hashes[exp.UPSTREAM_PATH.as_posix()].startswith("sha256:")
    assert sidecars[0]["counted_as_current"] is False

    copied = tmp_path / "upstream.json"
    bad = deepcopy(upstream)
    bad["law_fixture_ready_score"] = 0
    copied.write_text(json.dumps(bad), encoding="utf-8")
    bad_checks, _hashes, _value = exp.authenticate_upstream(copied)
    failed = next(row for row in bad_checks if row["artifact_field"] == "law_fixture_ready_score")
    assert failed["passed"] is False
    assert failed["expected"] == 1
    assert failed["observed"] == 0

    missing_checks, _hashes, _value = exp.authenticate_upstream(tmp_path / "missing.json")
    assert missing_checks[0]["observed"] == "missing"
    assert missing_checks[0]["passed"] is False

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    malformed_checks, _hashes, _value = exp.authenticate_upstream(malformed)
    assert malformed_checks[0]["observed"] == "malformed_json"

    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]", encoding="utf-8")
    nonobject_checks, _hashes, nonobject_value = exp.authenticate_upstream(nonobject)
    assert nonobject_value == {}
    assert any(row["passed"] is False for row in nonobject_checks)

    original_import = builtins.__import__

    def deny_rust_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "carnot" and "_rust" in fromlist:
            raise ImportError("unit missing extension")
        return original_import(name, globals, locals, fromlist, level)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", deny_rust_import)
        missing_rust_checks, _hashes, _upstream, _sidecars = exp.collect_preconditions(ROOT)
    rust_check = next(
        row for row in missing_rust_checks if row["check"] == "existing_rust_extension_available"
    )
    assert rust_check["passed"] is False


def test_terminal_reduction_preserves_complete_null_and_rejects_tampering() -> None:
    """SCENARIO-SAMPLER-7378-TERMINAL: raw rows determine scores and verdict."""

    artifact = exp.build_artifact_for_test(_passing_receipts())
    reduction = exp.independent_reduce(artifact)

    assert reduction["chain_evidence_complete"] is True
    assert reduction["source_cells_complete"] is True
    assert reduction["empty_support_cell_count"] == 18
    assert reduction["all_source_cells_qualified"] is False
    assert artifact["ising_sample_capture_complete_score"] == 1
    assert artifact["ising_law_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"]["current"] == exp.ZERO_CURRENT_INVOCATIONS
    assert artifact["execution_venue"] == "host"
    assert artifact["prior_attempt_findings"][0]["finding"] == "EXECUTION_VENUE_INVALID"
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["per_chain_results"].pop()
    assert "stored_reduction_mismatch" in exp.validate_artifact(changed)
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["ising_law_value_score"] = 1
    assert "score_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_incomplete" in exp.validate_artifact(changed)


def test_terminal_classifier_separates_blocked_disqualified_null_and_value() -> None:
    """REQ-SAMPLER-7378-SCORES: completion never substitutes for value."""

    baseline = {
        "prerequisites_passed": True,
        "required_validation_passed": True,
        "capture_complete": True,
        "science_qualified": False,
        "flagged_adversarial": False,
    }
    assert exp.classify_terminal(**baseline)["verdict_class"] == "null"
    assert exp.classify_terminal(**baseline)["ising_sample_capture_complete_score"] == 1
    assert exp.classify_terminal(**baseline)["ising_law_value_score"] == 0
    assert (
        exp.classify_terminal(**{**baseline, "science_qualified": True})["verdict_class"]
        == "circular_positive"
    )
    assert (
        exp.classify_terminal(**{**baseline, "required_validation_passed": False})["verdict_class"]
        == "disqualified"
    )
    assert (
        exp.classify_terminal(**{**baseline, "prerequisites_passed": False})["verdict_class"]
        == "blocked"
    )
    assert (
        exp.classify_terminal(**{**baseline, "flagged_adversarial": True})["verdict_class"]
        == "disqualified"
    )


def test_validation_plan_uses_scoped_commands_and_existing_e2e_harness(tmp_path: Path) -> None:
    """REQ-SAMPLER-7378-VALIDATION: Exp7358 and Exp7303 own command execution."""

    commands = exp.build_validation_plan(ROOT, tmp_path)
    names = [command.name for command in commands]

    assert names[: len(exp.REQUIRED_CHECK_NAMES)] == list(exp.REQUIRED_CHECK_NAMES)
    assert names[-2:] == list(exp.E2E_CHECK_NAMES)
    assert "full_python_suite" not in names
    assert any(
        any("test_e2e_training_sampling.py" in argument for argument in command.argv)
        for command in commands
    )
    assert any(
        any("test_ising_rust_python_energy_agreement" in argument for argument in command.argv)
        for command in commands
    )
    assert all("tests/python" not in command.argv for command in commands)
    assert exp.validate_validation_plan(ROOT, commands) == []
    assert (tmp_path / "basetemp").is_dir()
    assert (tmp_path / "coverage").is_dir()

    missing = commands[:-1]
    assert "validation_command_roster" in exp.validate_validation_plan(ROOT, missing)
    broad = [
        *commands,
        exp.validation_scope.CommandSpec(
            "bad_broad", (".venv/bin/pytest", "tests/python"), "forbidden"
        ),
    ]
    broad_errors = exp.validate_validation_plan(ROOT, broad)
    assert "broad_pytest_target" in broad_errors
    absent_parent = tmp_path / "does-not-exist" / "child"
    changed = list(commands)
    changed[-1] = exp.validation_scope.CommandSpec(
        changed[-1].name,
        (*changed[-1].argv, f"--basetemp={absent_parent}"),
        changed[-1].scope,
    )
    assert any(
        error.startswith("missing_basetemp_parent")
        for error in exp.validate_validation_plan(ROOT, changed)
    )


def test_validator_rejects_invalid_identity_and_blocked_work() -> None:
    """SCENARIO-SAMPLER-7378-TERMINAL: invalid terminal shapes fail closed."""

    artifact = exp.build_artifact_for_test(_passing_receipts())
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("run_date", "20260918", "identity_mismatch"),
        ("verdict_class", "success", "verdict_class_invalid"),
        ("MODEL_SPECS", ["forbidden"], "current_model_declaration_invalid"),
        ("model_invoked", True, "current_model_declaration_invalid"),
        ("invocation_counts", {"current": {}}, "current_invocation_counts_invalid"),
        ("inference_substrate_class", "gpu", "substrate_class_invalid"),
        ("execution_venue", "board", "execution_venue_invalid"),
        ("promotion_score", 1, "promotion_nonzero"),
        ("verifier_is_oracle", False, "circularity_declaration_invalid"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact_for_test()
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"] is not None
    assert blocked["per_chain_results"] == []
    assert blocked["ising_sample_capture_complete_score"] == 0
    assert blocked["ising_law_value_score"] == 0
    assert exp.validate_artifact(blocked) == []

    blocked_with_work = deepcopy(blocked)
    blocked_with_work["rows"] = [{"outcome": "forbidden"}]
    assert "blocked_artifact_has_dependent_work" in exp.validate_artifact(blocked_with_work)
    blocked_without_failure = deepcopy(blocked)
    blocked_without_failure["gate_check_summary"]["first_failure"] = None
    assert "blocked_gate_summary_missing" in exp.validate_artifact(blocked_without_failure)
    blocked_with_score = deepcopy(blocked)
    blocked_with_score["ising_law_value_score"] = 1
    assert "blocked_scores_nonzero" in exp.validate_artifact(blocked_with_score)

    flagged = deepcopy(artifact)
    flagged["flagged_adversarial"] = True
    flagged["ising_law_value_score"] = 1
    assert "adversarial_value_nonzero" in exp.validate_artifact(flagged)


def test_exact_observable_moments_are_bounded_and_normalized() -> None:
    """REQ-SAMPLER-7378-OBSERVABLES: exact targets cover energy and every bit."""

    formula = _fixture()
    target = exp.enumerated_target(formula, 2.0)
    names = [row["observable"] for row in target["observables"]]

    assert names == ["energy_per_clause", *[f"bit_{index}" for index in range(6)]]
    assert math.fsum(target["state_probabilities"].values()) == pytest.approx(1.0)
    assert all(0.0 <= row["mean"] <= 1.0 for row in target["observables"])
    assert all(row["variance"] >= 0.0 for row in target["observables"])
    assert target["independent_energy_parity_max_error"] <= 1e-12


def test_sampling_panel_walks_full_frozen_roster_without_copying_launchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-7378-PROTOCOL: orchestration visits every frozen arm once."""

    upstream = json.loads((ROOT / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))
    formulas = upstream["frozen_formulas"]
    monkeypatch.setattr(exp, "BETA_GRID", (1.0,))
    monkeypatch.setattr(exp, "CHAIN_SEEDS", (1, 2, 3, 4))
    monkeypatch.setattr(exp, "progress", lambda *args, **kwargs: None)

    def fake_target(
        formula: dict[str, object],
        beta: float,
        *,
        clauses: object = None,
    ) -> dict[str, object]:
        del beta, clauses
        return {
            "formula_id": formula["formula_id"],
            "support_size": 1,
            "state_probabilities": {"0": 1.0},
            "observables": [{"observable": "bit_0", "mean": 0.0, "variance": 0.0}],
            "target_sha256": "sha256:target",
        }

    def fake_chain(
        formula: dict[str, object],
        *,
        beta: float,
        condition: str,
        chain_order: int,
        seed: int,
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        del kwargs
        cell_id = f"{formula['formula_id']}|{beta}|{condition}"
        row = {
            "cell_id": cell_id,
            "formula_id": formula["formula_id"],
            "condition": condition,
            "beta": beta,
            "chain_order": chain_order,
            "seed": seed,
            "recorded_samples": 1,
            "state_counts": {"0": 1},
            "observable_series": {"bit_0": [0.0]},
            "outcome": "complete",
        }
        trace = (
            None
            if formula["formula_id"] == formulas[0]["formula_id"] and chain_order == 0
            else {
                "cell_id": cell_id,
                "chain_order": chain_order,
                "state_indices": [0],
            }
        )
        return row, trace

    def fake_cell(
        formula_id: str,
        beta: float,
        condition: str,
        chains: object,
        target: object,
    ) -> dict[str, object]:
        del chains, target
        return {
            "cell_id": f"{formula_id}|{beta}|{condition}",
            "formula_id": formula_id,
            "condition": condition,
            "cell_kind": (
                "wrong_law_negative_control"
                if condition == exp.NEGATIVE_CONDITION
                else "source_faithful"
            ),
            "qualified": True,
            "max_observable_error": 0.0,
            "observable_results": [{"observable": "bit_0", "observed_mean": 0.0}],
        }

    monkeypatch.setattr(exp, "enumerated_target", fake_target)
    monkeypatch.setattr(exp, "run_sampler_chain", fake_chain)
    monkeypatch.setattr(exp, "aggregate_cell", fake_cell)
    monkeypatch.setattr(
        exp,
        "distribution_shift_control",
        lambda formula, beta, source, appended: {
            "formula_id": formula["formula_id"],
            "beta": beta,
            "wrong_law_detected": True,
        },
    )

    chains, cells, controls, traces = exp.run_sampling_panel(formulas, started=0.0)
    assert len(chains) == (24 * 2 + 3) * 4
    assert len(cells) == 24 * 2 + 3
    assert len(controls) == 3
    assert len(traces) < len(chains)
    assert all("observable_series" not in row for row in chains)
