"""Tests for the independent exact certificate stream.

Spec refs: REQ-VERIFY-6744, SCENARIO-VERIFY-6744-GENERATION,
SCENARIO-VERIFY-6744-CERTIFICATES, SCENARIO-VERIFY-6744-RELABEL,
SCENARIO-VERIFY-6744-SPLIT, and SCENARIO-VERIFY-6744-REPLAY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6744_hardness_controlled_certificate_stream as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the exact stream once because every test reads immutable rows."""

    output = tmp_path_factory.mktemp("exp6744") / "stream.json"
    artifact = exp.run(output_path=output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    return artifact


def test_req_verify_6744_spec_and_preregistration_are_exact() -> None:
    """REQ-VERIFY-6744 fixes all counts before any solver call."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6744") :]
    for marker in (
        "SCENARIO-VERIFY-6744-GENERATION",
        "SCENARIO-VERIFY-6744-CERTIFICATES",
        "SCENARIO-VERIFY-6744-RELABEL",
        "SCENARIO-VERIFY-6744-SPLIT",
        "SCENARIO-VERIFY-6744-REPLAY",
        exp.INFERENCE_SUBSTRATE,
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section

    manifest = exp.build_preregistered_manifest()
    assert manifest["frozen_before_solving"] is True
    assert manifest["row_count"] == 72
    assert manifest["pair_count"] == 36
    assert manifest["seeds"] == list(exp.SEEDS)
    assert len(set(manifest["seeds"])) >= 3
    assert len(manifest["base_rows"]) == 36
    assert manifest["family_counts"] == {
        family: {"SAT": 12, "UNSAT": 12, "total": 24} for family in exp.FAMILIES
    }


@pytest.mark.parametrize("family", exp.FAMILIES)
@pytest.mark.parametrize("label", exp.LABELS)
@pytest.mark.parametrize("size_bin", tuple(exp.SIZE_BINS))
def test_scenario_6744_generators_and_independent_certificates(
    family: str, label: str, size_bin: str
) -> None:
    """SCENARIO-VERIFY-6744-CERTIFICATES checks both label classes."""

    cnf = exp.generate_formula(family, size_bin, label, exp.SEEDS[0])
    solved = exp.solve_cnf_exact(cnf)
    certificate = exp.make_certificate(cnf, solved)

    assert solved["label"] == label
    assert exp.check_certificate(cnf, label, certificate) is True
    assert exp.formula_graph_invariants(cnf)["variable_count"] == cnf["n_vars"]
    assert exp.canonical_clause_multiset(cnf["clauses"])

    corrupted = deepcopy(certificate)
    if label == "SAT":
        variable = next(iter(corrupted["assignment"]))
        corrupted["assignment"][variable] = not corrupted["assignment"][variable]
    else:
        corrupted["falsified_clause_by_assignment"][0] = len(cnf["clauses"])
    assert exp.check_certificate(cnf, label, corrupted) is False


def test_scenario_6744_generator_and_certificate_edges_fail_closed() -> None:
    """SCENARIO-VERIFY-6744-CERTIFICATES rejects malformed exact inputs."""

    with pytest.raises(ValueError, match="unknown_family"):
        exp.generate_formula("unknown", "small", "SAT", exp.SEEDS[0])
    with pytest.raises(ValueError, match="unknown_size_bin"):
        exp.generate_formula(exp.FAMILIES[0], "huge", "SAT", exp.SEEDS[0])
    with pytest.raises(ValueError, match="unknown_label"):
        exp.generate_formula(exp.FAMILIES[0], "small", "MAYBE", exp.SEEDS[0])

    sat_cnf = exp.generate_formula(exp.FAMILIES[0], "small", "SAT", exp.SEEDS[0])
    sat = exp.make_certificate(sat_cnf, exp.solve_cnf_exact(sat_cnf))
    assert exp.check_certificate(sat_cnf, "SAT", {}) is False
    missing_assignment = deepcopy(sat)
    missing_assignment["assignment"].pop("1")
    assert exp.check_certificate(sat_cnf, "SAT", missing_assignment) is False
    non_boolean = deepcopy(sat)
    non_boolean["assignment"]["1"] = 1
    assert exp.check_certificate(sat_cnf, "SAT", non_boolean) is False
    assert exp.check_certificate(sat_cnf, "UNKNOWN", sat) is False

    unsat_cnf = exp.generate_formula(exp.FAMILIES[0], "small", "UNSAT", exp.SEEDS[0])
    unsat = exp.make_certificate(unsat_cnf, exp.solve_cnf_exact(unsat_cnf))
    wrong_count = deepcopy(unsat)
    wrong_count["assignment_count"] = 0
    assert exp.check_certificate(unsat_cnf, "UNSAT", wrong_count) is False
    short_cover = deepcopy(unsat)
    short_cover["falsified_clause_by_assignment"].pop()
    assert exp.check_certificate(unsat_cnf, "UNSAT", short_cover) is False
    wrong_witness = deepcopy(unsat)
    wrong_witness["falsified_clause_by_assignment"][0] = next(
        index
        for index, clause in enumerate(unsat_cnf["clauses"])
        if exp._clause_is_satisfied(
            clause, {variable: False for variable in range(1, unsat_cnf["n_vars"] + 1)}
        )
    )
    assert exp.check_certificate(unsat_cnf, "UNSAT", wrong_witness) is False


def test_scenario_6744_relabel_pairs_preserve_all_invariants(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6744-RELABEL verifies every proof-preserving pair."""

    rows = ready_artifact["rows"]
    by_id = {row["row_id"]: row for row in rows}
    receipts = ready_artifact["relabel_pair_receipts"]

    assert len(rows) == 72
    assert len(receipts) == 36
    assert all(receipt["passed"] for receipt in receipts)
    for receipt in receipts:
        base = by_id[receipt["base_row_id"]]
        mate = by_id[receipt["mate_row_id"]]
        assert base["pair_id"] == mate["pair_id"]
        assert base["split"] == mate["split"]
        assert base["label"] == mate["label"]
        assert exp.verify_relabel_pair(base, mate)["passed"] is True


def test_scenario_6744_family_splits_and_work_metadata_are_isolated(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6744-SPLIT keeps work counters diagnostic only."""

    manifest = ready_artifact["split_manifest"]
    assert manifest["family_disjoint"] is True
    assert manifest["pair_leak_count"] == 0
    assert manifest["row_assignment_complete"] is True
    split_families = [set(row["families"]) for row in manifest["splits"].values()]
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(split_families)
        for right in split_families[i + 1 :]
    )

    metadata = ready_artifact["solver_work_metadata"]
    assert len(metadata) == 72
    assert all(
        set(row) == {"row_id", "conflicts", "decisions", "propagations", "wall_time_s"}
        for row in metadata
    )
    assert all(row["wall_time_s"] >= 0 for row in metadata)
    assert "model_hard" not in exp.canonical_json(ready_artifact)


def test_scenario_6744_ready_artifact_replays_and_explains_every_field(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6744-REPLAY admits only a complete exact stream."""

    assert exp.validate_artifact(ready_artifact) == []
    assert ready_artifact["status"] == "complete_ready"
    assert ready_artifact["hardness_stream_ready"] is True
    assert ready_artifact["verdict_class"] == "positive"
    assert ready_artifact["honest_verdict"].startswith("complete_positive")
    assert ready_artifact["duration_s"] > 0
    assert ready_artifact["random_seed"] == list(exp.SEEDS)
    assert ready_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(ready_artifact["field_principles"]) == set(ready_artifact)
    assert ready_artifact["reproducibility_checksum"] == exp.stream_checksum(ready_artifact["rows"])
    assert ready_artifact["deterministic_replay_receipt"]["matched"] is True
    assert ready_artifact["gate_check_summary"] == {
        "failed_check": None,
        "expected_value": True,
        "observed_value": True,
    }
    assert all(
        receipt["solver_exit_code"] == 0
        for receipt in ready_artifact["certificate_checker_receipts"]
    )
    assert all(
        receipt["checker_exit_code"] == 0
        for receipt in ready_artifact["certificate_checker_receipts"]
    )
    assert all(
        receipt["solver_sha256"] != receipt["checker_sha256"]
        for receipt in ready_artifact["certificate_checker_receipts"]
    )


def test_scenario_6744_adversarial_mutations_fail_closed(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6744-REPLAY catches row, split, pair, and gate drift."""

    cases: list[tuple[str, dict[str, object]]] = []

    bad_certificate = deepcopy(ready_artifact)
    bad_certificate["rows"][0]["certificate_sha256"] = "sha256:bad"
    cases.append(("row_certificate_invalid", bad_certificate))

    bad_pair = deepcopy(ready_artifact)
    bad_pair["relabel_pair_receipts"][0]["passed"] = False
    cases.append(("pair_receipt_invalid", bad_pair))

    bad_split = deepcopy(ready_artifact)
    bad_split["rows"][0]["split"] = "test"
    cases.append(("split_manifest_invalid", bad_split))

    bad_gate = deepcopy(ready_artifact)
    bad_gate["hardness_stream_ready"] = False
    cases.append(("readiness_recomputation_mismatch", bad_gate))

    for expected, artifact in cases:
        errors = exp.validate_artifact(artifact)
        assert expected in errors

    assert exp.validate_artifact({})[0].startswith("missing_required_fields:")


def test_scenario_6744_reducer_and_schema_edges_fail_closed(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6744-REPLAY covers every readiness reduction gate."""

    mutations = []
    too_short = deepcopy(ready_artifact)
    too_short["rows"].pop()
    mutations.append(too_short)
    bad_registered_counts = deepcopy(ready_artifact)
    bad_registered_counts["preregistered_manifest"]["family_counts"] = {}
    mutations.append(bad_registered_counts)
    bad_observed_counts = deepcopy(ready_artifact)
    bad_observed_counts["family_counts"] = {}
    bad_observed_counts["preregistered_manifest"]["family_counts"] = {}
    mutations.append(bad_observed_counts)
    failed_checker = deepcopy(ready_artifact)
    failed_checker["certificate_checker_receipts"][0]["passed"] = False
    mutations.append(failed_checker)
    missing_checker = deepcopy(ready_artifact)
    missing_checker["certificate_checker_receipts"].pop()
    mutations.append(missing_checker)
    missing_pair = deepcopy(ready_artifact)
    missing_pair["relabel_pair_receipts"].pop()
    mutations.append(missing_pair)
    failed_split = deepcopy(ready_artifact)
    failed_split["split_manifest"] = {}
    mutations.append(failed_split)
    split_not_complete = deepcopy(ready_artifact)
    split_not_complete["rows"][0]["split"] = "test"
    split_not_complete["split_manifest"] = exp.build_split_manifest(split_not_complete["rows"])
    mutations.append(split_not_complete)
    bad_replay = deepcopy(ready_artifact)
    bad_replay["deterministic_replay_receipt"]["matched"] = False
    mutations.append(bad_replay)

    assert all(exp._ready_recomputation(artifact) is False for artifact in mutations)

    schema_cases = {
        "field_principles_missing": ("field_principles", {}),
        "inference_substrate_mismatch": ("inference_substrate", "llm"),
        "random_seed_mismatch": ("random_seed", []),
        "duration_not_positive": ("duration_s", 0),
        "ready_verdict_class_invalid": ("verdict_class", "null"),
        "ready_verdict_prefix_invalid": ("honest_verdict", "complete_null"),
    }
    for error, (field, value) in schema_cases.items():
        mutated = deepcopy(ready_artifact)
        mutated[field] = value
        assert error in exp.validate_artifact(mutated)

    blocked = exp._blocked_artifact(
        [{"check": "x", "expected_value": True, "observed_value": False, "passed": False}],
        exp.build_preregistered_manifest(),
        1.0,
    )
    blocked["rows"] = [{}]
    blocked["hardness_stream_ready"] = True
    blocked["gate_check_summary"] = {"failed_check": None}
    blocked["honest_verdict"] = "blocked"
    assert {
        "blocked_rows_or_readiness_present",
        "blocked_gate_summary_missing",
        "blocked_verdict_prefix_invalid",
    } <= set(exp.validate_artifact(blocked))


def test_req_6744_blocked_checker_precondition_is_complete(tmp_path: Path) -> None:
    """REQ-VERIFY-6744 emits the owned blocked verdict without partial rows."""

    checks = exp.collect_preconditions(tmp_path / "blocked.json")
    checks[0]["observed_value"] = False
    checks[0]["passed"] = False
    artifact = exp.run(
        output_path=tmp_path / "blocked.json",
        precondition_probe=lambda _path: checks,
    )

    assert artifact["hardness_stream_ready"] is False
    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_exact_checker")
    assert artifact["gate_check_summary"]["failed_check"] == checks[0]["check"]
    assert exp.validate_artifact(artifact) == []

    nested = tmp_path / "absent" / "deeper" / "stream.json"
    assert exp.collect_preconditions(nested)[-1]["passed"] is True

    original = exp.collect_preconditions
    try:
        exp.collect_preconditions = lambda _path: checks
        direct = exp.build_artifact(tmp_path / "direct-blocked.json", 1.0)
    finally:
        exp.collect_preconditions = original
    assert direct["verdict_class"] == "blocked"


def test_req_6744_atomic_writer_and_cli_validation(
    tmp_path: Path, ready_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6744 writes atomically and exposes replay validation."""

    output = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(output, ready_artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == ready_artifact
    assert list(output.parent.glob("*.tmp")) == []

    parsed = exp.parse_args(["--output", str(output), "--validate"])
    assert parsed.output == output
    assert parsed.validate is True
    assert exp.main(["--output", str(output), "--validate"]) == 0

    output.write_text(json.dumps({"bad": True}), encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 1
    output.write_text("{", encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 1
    assert exp.main(["--output", str(tmp_path / "missing.json"), "--validate"]) == 1

    generated = tmp_path / "main-generated.json"
    assert exp.main(["--output", str(generated)]) == 0
    assert generated.exists()

    real_build = exp.build_artifact
    try:
        exp.build_artifact = lambda _path, _duration: {
            **deepcopy(ready_artifact),
            "inference_substrate": "llm",
        }
        with pytest.raises(ValueError, match="invalid_exp6744_artifact"):
            exp.run(tmp_path / "invalid-run.json")
    finally:
        exp.build_artifact = real_build

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(exp.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        exp.write_json_atomic(tmp_path / "failed.json", ready_artifact)
    assert list(tmp_path.glob("*.tmp")) == []
