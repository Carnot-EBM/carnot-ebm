"""Tests for Exp5761 exact constraint-acquisition benchmark.

Spec refs: REQ-BENCH-5761, REQ-LEARN-5761, REQ-STORE-5761,
SCENARIO-BENCH-5761, SCENARIO-BENCH-5761-CONTROLS,
SCENARIO-LEARN-5761, SCENARIO-LEARN-5761-MINIMAL-QUERIES,
SCENARIO-STORE-5761.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5761_exact_constraint_acquisition_benchmark as mod


REPO = Path(__file__).resolve().parents[2]
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5761_exact_constraint_acquisition_benchmark.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py "
    "-m pytest tests/python/test_experiment_5761_exact_constraint_acquisition_benchmark.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5761_exact_constraint_acquisition_benchmark.json"
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
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _run_fixture(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        benchmark_manifest_path=tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name,
        preconditions_checked=mod.fixture_preconditions(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_5761_specs_declare_acquisition_contract() -> None:
    """REQ-BENCH-5761/REQ-LEARN-5761/REQ-STORE-5761: OpenSpec anchors Exp5761."""

    bench = BENCH_SPEC.read_text(encoding="utf-8")
    learn = LEARN_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    bench_section = bench[bench.index("### REQ-BENCH-5761") : bench.index("### REQ-BENCH-3389")]
    learn_section = learn[learn.index("## REQ-LEARN-5761") : learn.index("## REQ-LEARN-5737")]
    store_section = store[store.index("### REQ-STORE-5761") :]

    for marker in (
        "REQ-BENCH-5761",
        "SCENARIO-BENCH-5761-CONTROLS",
        str(mod.RESULT_RELATIVE_PATH),
        "`ca_benchmark_ready_score`",
        "`exact_validator_disagreement_count`",
        "`train_dev_science_disjoint_score`",
        "deterministic_exact_solver_dataset_generation_no_llm",
    ):
        assert marker in bench_section
    for marker in (
        "REQ-LEARN-5761",
        "SCENARIO-LEARN-5761-MINIMAL-QUERIES",
        "add_missing_constraint",
        "remove_spurious_constraint",
    ):
        assert marker in learn_section
    for marker in ("REQ-STORE-5761", "SCENARIO-STORE-5761", "model AST from model text"):
        assert marker in store_section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in bench_section


def test_scenario_5761_generates_balanced_sealed_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5761: rows, splits, variants, and gate scalars are sealed."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)
    rerun = _run_fixture(tmp_path)

    assert artifact == rerun
    assert mod.validate_artifact(artifact) is True
    assert mod.verify_benchmark_manifest(rows, artifact) is True
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["instance_count"] == mod.INSTANCE_COUNT
    assert artifact["family_counts"] == {family: 30 for family in mod.REQUIRED_FAMILIES}
    assert artifact["variant_counts"] == {kind: mod.INSTANCE_COUNT for kind in mod.VARIANT_KINDS}
    assert artifact["split_manifest"]["split_counts"] == {"dev": 40, "science": 40, "train": 40}
    for split_counts in artifact["split_manifest"]["family_counts"].values():
        assert split_counts == {family: 10 for family in mod.REQUIRED_FAMILIES}
    assert len(artifact["science_row_hashes"]) == 40
    assert artifact["positive_assignment_count"] == mod.INSTANCE_COUNT * len(mod.VARIANT_KINDS)
    assert artifact["negative_assignment_count"] == mod.INSTANCE_COUNT * len(mod.VARIANT_KINDS)
    assert artifact["membership_query_count"] == mod.INSTANCE_COUNT * 4
    assert artifact["structure_receipt_failure_count"] == 0
    assert artifact["solution_receipt_failure_count"] == 0
    assert artifact["exact_validator_disagreement_count"] == 0
    assert artifact["train_dev_science_disjoint_score"] == pytest.approx(1.0)
    assert artifact["ca_benchmark_ready_score"] == pytest.approx(1.0)
    assert artifact["llm_inference_used"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)
    assert artifact["benchmark_manifest_hash"] == mod.sha256_file(
        tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact


def _variant(row: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(item for item in row["variants"] if item["variant_kind"] == kind)


def test_req_5761_variants_have_repair_receipts_and_minimal_queries(tmp_path: Path) -> None:
    """REQ-LEARN-5761: each variant maps to exact add/remove/noop repairs."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)

    for family in mod.REQUIRED_FAMILIES:
        row = next(item for item in rows if item["family"] == family)
        assert row["row_hash"] == artifact["benchmark_row_hashes"][row["case_id"]]
        assert set(row["variants_by_kind"]) == set(mod.VARIANT_KINDS)
        assert artifact["domain_artifact_hashes"][row["case_id"]] == row["domain_artifact_hash"]

        faithful = _variant(row, "faithful")
        incomplete = _variant(row, "incomplete")
        overfit = _variant(row, "overfit")
        mixed = _variant(row, "mixed")

        assert artifact["faithful_model_hashes"][row["case_id"]] == faithful["model_hash"]
        assert artifact["incomplete_model_hashes"][row["case_id"]] == incomplete["model_hash"]
        assert artifact["overfit_model_hashes"][row["case_id"]] == overfit["model_hash"]
        assert artifact["mixed_model_hashes"][row["case_id"]] == mixed["model_hash"]

        expected_operations = {
            "faithful": ["noop"],
            "incomplete": ["add_missing_constraint"],
            "overfit": ["remove_spurious_constraint"],
            "mixed": ["add_missing_constraint", "remove_spurious_constraint"],
        }
        for variant in (faithful, incomplete, overfit, mixed):
            variant_id = variant["variant_id"]
            repair = artifact["expected_repair_receipts"][variant_id]
            query = artifact["distinguishing_query_receipts"][variant_id]
            role = artifact["hard_soft_role_receipts"][variant_id]

            assert repair["operation_types"] == expected_operations[variant["variant_kind"]]
            assert repair["expected_repair_hash"] == variant["expected_repair_hash"]
            assert query["query_hash"] == variant["distinguishing_query_hash"]
            assert query["minimal"] is True
            assert role["soft_preference_ids_unchanged"] is True
            assert role["soft_objective_unchanged"] is True
            assert variant["positive_assignment_receipt"]["accepted_by_variant"] is True
            assert variant["negative_assignment_receipt"]["accepted_by_variant"] is False

        assert incomplete["distinguishing_query_receipt"]["query_count"] == 1
        assert incomplete["distinguishing_query_receipt"]["directions"] == ["variant_accepts_extra"]
        assert overfit["distinguishing_query_receipt"]["query_count"] == 1
        assert overfit["distinguishing_query_receipt"]["directions"] == ["variant_rejects_faithful"]
        assert mixed["distinguishing_query_receipt"]["query_count"] == 2
        assert sorted(mixed["distinguishing_query_receipt"]["directions"]) == [
            "variant_accepts_extra",
            "variant_rejects_faithful",
        ]


def test_req_5761_preconditions_and_adversarial_controls_fail_closed(tmp_path: Path) -> None:
    """REQ-BENCH-5761: preflight and invalid evidence controls block readiness."""

    preconditions = mod.collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )
    instances = mod.generate_instances(preconditions)
    controls = mod.build_adversarial_controls(instances)
    blocked = mod.fixture_preconditions()
    blocked["preconditions_ready"] = False
    blocked["faithful_structure_reconstruction"]["ok"] = False
    blocked["blocked_reasons"] = ["faithful_model_structure_reconstruction_failed"]
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        benchmark_manifest_path=tmp_path / "blocked.jsonl",
        preconditions_checked=blocked,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert preconditions["preconditions_ready"] is True
    assert preconditions["exp5746_replay"]["ok"] is True
    assert preconditions["license_provenance"]["mpmmine_result_imported"] is False
    assert preconditions["faithful_structure_reconstruction"]["ok"] is True
    assert {row["case_id"] for row in instances} == set(artifact["benchmark_row_hashes"]) or artifact[
        "status"
    ] == "blocked"
    assert set(controls) == set(mod.ADVERSARIAL_CONTROL_TYPES)
    assert all(control["detected"] is True for control in controls.values())
    assert artifact["status"] == "blocked"
    assert artifact["ca_benchmark_ready_score"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.read_benchmark_manifest(tmp_path / "blocked.jsonl") == []
    assert mod.validate_artifact(artifact) is True


def test_req_5761_validation_and_manifest_tamper_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5761-CONTROLS: schema, checksum, and manifest tamper fail."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)

    missing = deepcopy(artifact)
    del missing["verifier_is_oracle"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_score = deepcopy(artifact)
    bad_score["ca_benchmark_ready_score"] = 0.0
    bad_score["honest_verdict"] = mod.honest_verdict(bad_score)
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ca_benchmark_ready_score"):
        mod.validate_artifact(bad_score)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_llm = deepcopy(artifact)
    bad_llm["llm_inference_used"] = True
    bad_llm["ca_benchmark_ready_score"] = mod.ca_benchmark_ready_score(bad_llm)
    bad_llm["honest_verdict"] = mod.honest_verdict(bad_llm)
    bad_llm["reproducibility_checksum"] = mod.reproducibility_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_inference_used"):
        mod.validate_artifact(bad_llm)

    reasons = mod.blocked_reasons(
        {
            **artifact,
            "adversarial_controls": {
                **artifact["adversarial_controls"],
                "shortcut": {**artifact["adversarial_controls"]["shortcut"], "detected": False},
            },
            "train_dev_science_disjoint_score": 0.0,
            "exact_validator_disagreement_count": 1,
        }
    )
    assert "adversarial_control_not_detected" in reasons
    assert "train_dev_science_disjointness_failed" in reasons
    assert "exact_validator_disagreement_count" in reasons

    tampered_rows = deepcopy(rows)
    tampered_rows[0]["row_hash"] = "sha256:" + "1" * 64
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_benchmark_manifest(tampered_rows, artifact)

    bad_artifact = deepcopy(artifact)
    bad_artifact["benchmark_manifest_hash"] = "sha256:" + "2" * 64
    with pytest.raises(mod.ManifestReplayError, match="benchmark_manifest_hash"):
        mod.verify_benchmark_manifest(rows, bad_artifact)


def test_req_5761_defensive_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-BENCH-5761: malformed evidence and defensive validators fail closed."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)
    source = mod.exp5746.read_benchmark_manifest(
        REPO / mod.exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH
    )[0]
    source_assignment = source["candidate_pool"][0]["assignment"]
    source_ast = source["canonical_typed_formulation"]

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json_object(list_json)

    broken_source = deepcopy(source)
    broken_source["row_hash"] = "sha256:" + "0" * 64
    reconstruction = mod._reconstruct_faithful_receipts([broken_source])
    assert reconstruction["ok"] is False
    assert reconstruction["failure_count"] == 1

    missing_preflight = mod.collect_preconditions(
        upstream_artifact_path=tmp_path / "missing.json",
        upstream_manifest_path=tmp_path / "missing.jsonl",
        memory_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
        disk_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
    )
    assert missing_preflight["preconditions_ready"] is False
    for reason in (
        "exp5746_replay_failed",
        "insufficient_free_ram",
        "insufficient_free_disk",
        "family_balance_failed",
        "deterministic_seed_replay_failed",
        "faithful_model_structure_reconstruction_failed",
        "license_or_provenance_failed",
    ):
        assert reason in missing_preflight["blocked_reasons"]

    forbid = {"id": "forbid-exact", "type": "forbid_assignment", "assignment": source_assignment}
    assert mod._constraint_holds(source_ast, forbid, source_assignment) is False
    with pytest.raises(ValueError, match="unsupported constraint type"):
        mod._constraint_holds(source_ast, {"id": "bad", "type": "bad"}, source_assignment)

    bad_soft_ast = deepcopy(source_ast)
    bad_soft_ast["soft_preferences"] = [{"id": "bad-soft", "type": "bad_soft"}]
    with pytest.raises(ValueError, match="unsupported soft preference type"):
        mod._objective_value(source, bad_soft_ast, source_assignment)

    empty_source = deepcopy(source)
    empty_source["candidate_pool"] = []
    with pytest.raises(ValueError, match="no non-equivalent acquisition mutations"):
        mod._choose_mutations(empty_source, source_ast)

    disagreed = deepcopy(rows[:1])
    disagreed[0]["variants"][0]["solution_receipt"]["optimum_value"] = -999
    failures = mod.collect_independent_validator_failures(disagreed)
    assert failures["exact_validator_disagreement_count"] == 1
    assert mod.build_adversarial_controls([]) == {}

    bad_structure = deepcopy(rows[:1])
    bad_structure[0]["variants"][0]["hard_soft_role_receipt"][
        "soft_objective_unchanged"
    ] = False
    bad_structure[0]["variants"][1]["mutation_receipt"]["semantic_change"] = False
    assert mod._structure_receipt_failure_count(bad_structure) == 2

    bad_solution = deepcopy(rows[:1])
    bad_variant = bad_solution[0]["variants"][0]
    bad_variant["solution_receipt"]["satisfiable"] = False
    bad_variant["positive_assignment_receipt"]["accepted_by_variant"] = False
    bad_variant["negative_assignment_receipt"]["accepted_by_variant"] = True
    bad_variant["distinguishing_query_receipt"]["minimal"] = False
    assert mod._solution_receipt_failure_count(bad_solution) == 4

    reasoned = deepcopy(artifact)
    reasoned["inference_substrate"] = "wrong"
    reasoned["benchmark_manifest_hash"] = ""
    reasons = mod.blocked_reasons(reasoned)
    assert "inference_substrate" in reasons
    assert "benchmark_manifest_hash" in reasons

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "wrong"
    bad_substrate["ca_benchmark_ready_score"] = mod.ca_benchmark_ready_score(bad_substrate)
    bad_substrate["status"] = "blocked"
    bad_substrate["honest_verdict"] = mod.honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_gate = deepcopy(artifact)
    bad_gate["producer_gate_fields"] = ["bad"]
    bad_gate["ca_benchmark_ready_score"] = mod.ca_benchmark_ready_score(bad_gate)
    bad_gate["status"] = "blocked"
    bad_gate["honest_verdict"] = mod.honest_verdict(bad_gate)
    bad_gate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_gate)
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(bad_gate)

    wrapped_gate = deepcopy(artifact)
    wrapped_gate["ca_benchmark_ready_score"] = {"value": 1.0}
    wrapped_gate["reproducibility_checksum"] = mod.reproducibility_checksum(wrapped_gate)
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(wrapped_gate)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_complete_verdict = deepcopy(artifact)
    bad_complete_verdict["honest_verdict"] = "blocked: wrong"
    bad_complete_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_complete_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_complete_verdict)

    bad_science = deepcopy(artifact)
    bad_science["science_row_hashes"] = []
    with pytest.raises(mod.ManifestReplayError, match="science_row_hashes"):
        mod.verify_benchmark_manifest(rows, bad_science)


def test_req_5761_mutation_search_skips_invalid_spurious_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-LEARN-5761-MINIMAL-QUERIES: no-op and infeasible additions are rejected."""

    source = mod.exp5746.read_benchmark_manifest(
        REPO / mod.exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH
    )[0]
    source_ast = source["canonical_typed_formulation"]
    original_additions = mod._addition_candidates(source, source_ast)
    first_variable = source_ast["variables"][0]
    rejected_candidate_id = source["exact_optimum_receipt"]["optimal_candidate_ids"][0]

    def fake_additions(
        source_row: dict[str, Any],
        faithful_ast: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del source_row, faithful_ast
        return [
            {
                "constraint": {
                    "id": "control-unsat-equals",
                    "type": "equals",
                    "var": first_variable["name"],
                    "value": "__absent_from_domain__",
                    "spurious": True,
                    "restriction_scope": "global",
                },
                "rejected_candidate_id": rejected_candidate_id,
            },
            {
                "constraint": {"id": "control-noop-vacuous", "type": "vacuous_global"},
                "rejected_candidate_id": rejected_candidate_id,
            },
            *original_additions,
        ]

    monkeypatch.setattr(mod, "_addition_candidates", fake_additions)
    mutations = mod._choose_mutations(source, source_ast)

    assert mutations["overfit"]["added_constraint"]["id"] not in {
        "control-unsat-equals",
        "control-noop-vacuous",
    }
