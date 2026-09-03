"""Tests for the deterministic exact reformulation fixture.

Spec refs: REQ-VERIFY-6955 and SCENARIO-VERIFY-6955-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6955_reformulation_fixture as exp


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pairs() -> list[dict[str, object]]:
    """Generate the corpus once for direct schema and proof checks."""

    return exp.generate_pairs(exp.RANDOM_SEED)


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the full artifact once because it starts a fresh replay process."""

    checkpoint = tmp_path_factory.mktemp("exp6955") / "corpus.json"
    return exp.build_artifact(date="20260903", repo_root=ROOT, checkpoint_path=checkpoint)


def test_req_verify_6955_spec_precedes_and_owns_the_contract() -> None:
    """REQ-VERIFY-6955 declares all fields and scenarios before implementation."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6955") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-VERIFY-6955-{name}" in section
        for name in (
            "MAPPING",
            "OBJECTIVE",
            "HARD-NEGATIVES",
            "AUTHORITY",
            "SPLITS",
            "REPLAY",
            "READINESS",
        )
    )


def test_req_verify_6955_corpus_has_exact_balance_and_small_domains(
    pairs: list[dict[str, object]],
) -> None:
    """REQ-VERIFY-6955 freezes 120 balanced, bounded pairs over three families."""

    labels = Counter(str(pair["expected_label"]) for pair in pairs)
    families = Counter(str(pair["family"]) for pair in pairs)
    splits = Counter(str(pair["split"]) for pair in pairs)

    assert len(pairs) == 120
    assert labels == {"equivalent": 72, "non_equivalent": 48}
    assert families == {family: 40 for family in exp.FAMILIES}
    assert set(splits) == set(exp.SPLITS)
    assert all(
        len(variable["universe"]) <= exp.MAX_UNIVERSE_SIZE
        for pair in pairs
        for side in ("source", "target")
        for variable in pair[side]["variables"]
    )


def test_scenario_verify_6955_mapping_accepts_affine_renaming_and_rejects_shape(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-MAPPING enforces exact keys and variable bijections."""

    positive = next(pair for pair in pairs if pair["expected_label"] == "equivalent")
    canonical = exp.canonical_mapping(positive["mapping"], positive["source"], positive["target"])
    source_names = {variable["name"] for variable in positive["source"]["variables"]}
    target_names = {variable["name"] for variable in positive["target"]["variables"]}

    assert canonical["schema_version"] == exp.MAPPING_SCHEMA_VERSION
    assert {row["source"] for row in canonical["variables"]} == source_names
    assert {row["target"] for row in canonical["variables"]} == target_names
    assert any(row["source"] != row["target"] for row in canonical["variables"])
    assert any(row["offset"] != "0" for row in canonical["variables"])

    variants: list[tuple[dict[str, object], str]] = []
    extra = deepcopy(positive["mapping"])
    extra["extra"] = True
    variants.append((extra, "mapping_keys"))
    missing = deepcopy(positive["mapping"])
    missing.pop("objective")
    variants.append((missing, "mapping_keys"))
    missing_variable = deepcopy(positive["mapping"])
    missing_variable["variables"].pop()
    variants.append((missing_variable, "source_variable_coverage"))
    duplicate = deepcopy(positive["mapping"])
    duplicate["variables"][1]["target"] = duplicate["variables"][0]["target"]
    variants.append((duplicate, "duplicate_target_variable"))
    nested_extra = deepcopy(positive["mapping"])
    nested_extra["variables"][0]["note"] = "not allowed"
    variants.append((nested_extra, "variable_mapping_keys"))

    for mapping, reason in variants:
        with pytest.raises(exp.MappingSchemaError, match=reason):
            exp.canonical_mapping(mapping, positive["source"], positive["target"])


def test_scenario_verify_6955_objective_scale_sign_and_direction(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-OBJECTIVE proves positive and negative rational scales."""

    positives = [pair for pair in pairs if pair["expected_label"] == "equivalent"]
    scales = {pair["mapping"]["objective"]["scale"] for pair in positives}
    negative = next(
        pair for pair in positives if exp.as_fraction(pair["mapping"]["objective"]["scale"]) < 0
    )
    enum = exp.prove_pair_with_enumerator(negative)
    z3_row = exp.prove_pair_with_z3(negative)

    assert {"1", "2", "1/2", "-1", "-2"} <= scales
    assert (
        negative["mapping"]["objective"]["source_direction"]
        != negative["mapping"]["objective"]["target_direction"]
    )
    assert enum["objective_affine_preserved"] is True
    assert enum["objective_order_preserved"] is True
    assert z3_row["label"] == "equivalent"
    assert z3_row["objective_affine_status"] == "unsat"
    assert z3_row["objective_order_status"] == "unsat"


def test_scenario_verify_6955_hard_negative_roster_has_one_edit_and_counterevidence(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-HARD-NEGATIVES retains each single-edit failure."""

    negatives = [pair for pair in pairs if pair["expected_label"] == "non_equivalent"]
    edits = Counter(str(pair["hard_negative_edit"]) for pair in negatives)

    assert set(edits) == set(exp.HARD_NEGATIVE_EDITS)
    assert all(pair["hard_negative_edit_count"] == 1 for pair in negatives)
    for edit in exp.HARD_NEGATIVE_EDITS:
        pair = next(row for row in negatives if row["hard_negative_edit"] == edit)
        enum = exp.prove_pair_with_enumerator(pair)
        z3_row = exp.prove_pair_with_z3(pair)
        assert enum["label"] == "non_equivalent"
        assert enum["counterexample"] is not None
        assert z3_row["label"] == "non_equivalent"


def test_scenario_verify_6955_both_exact_authorities_agree_on_every_pair(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-AUTHORITY proves all rows without unknown results."""

    for pair in pairs:
        enum = exp.prove_pair_with_enumerator(pair)
        z3_row = exp.prove_pair_with_z3(pair)
        assert enum["label"] == pair["expected_label"]
        assert z3_row["label"] == pair["expected_label"]
        assert z3_row["status"] in {"proved", "counterexample", "schema_rejected"}
        assert z3_row["status"] not in {"unknown", "timeout"}


def test_scenario_verify_6955_splits_are_template_frozen_and_isolated(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6955-SPLITS prevents normalized or isomorphic leakage."""

    split_rows = ready_artifact["split_rows"]
    template_splits: dict[str, set[str]] = {}
    for row in split_rows:
        template_splits.setdefault(row["generator_template"], set()).add(row["split"])

    assert all(len(split_set) == 1 for split_set in template_splits.values())
    assert {row["split"] for row in split_rows} == set(exp.SPLITS)
    assert all(not row["normalized_hash_crosses_split"] for row in split_rows)
    assert all(not row["isomorphism_hash_crosses_split"] for row in split_rows)
    assert all(not row["crosses_split"] for row in ready_artifact["isomorphism_rows"])


def test_scenario_verify_6955_ready_artifact_recomputes_all_required_rows(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6955-READINESS derives the gate from complete row evidence."""

    artifact = ready_artifact

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert len(artifact["rows"]) == len(artifact["pair_rows"]) == 120
    assert len(artifact["formulation_rows"]) == 240
    assert len(artifact["mapping_rows"]) == 120
    assert len(artifact["hard_negative_rows"]) == 48
    assert len(artifact["feasibility_witness_rows"]) == 120
    assert len(artifact["objective_order_rows"]) == 120
    assert len(artifact["z3_rows"]) == len(artifact["enumeration_rows"]) == 120
    assert len(artifact["authority_agreement_rows"]) == 120
    assert len(artifact["fresh_process_replay_rows"]) == 120
    assert all(row["authorities_agree"] for row in artifact["authority_agreement_rows"])
    assert all(row["replay_matches"] for row in artifact["fresh_process_replay_rows"])
    assert artifact["quarantine_rows"] == []
    assert artifact["reformulation_fixture_ready_score"] == 1
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_circular_positive")
    assert exp.payload_checksum(artifact) == artifact["reproducibility_checksum"]


def test_scenario_verify_6955_replay_is_deterministic_from_serialized_checkpoint(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6955-REPLAY recomputes every scientific identity."""

    checkpoint = ROOT / ready_artifact["corpus_checkpoint_path"]
    if not checkpoint.is_file():
        checkpoint = Path(ready_artifact["corpus_checkpoint_path"])
    first = exp.replay_checkpoint(checkpoint)
    second = exp.replay_checkpoint(checkpoint)

    assert first == second
    assert len(first) == 120
    assert all(row["label"] in {"equivalent", "non_equivalent"} for row in first)
    assert all(row["split"] in exp.SPLITS for row in first)
    assert all(row["pair_hash"].startswith("sha256:") for row in first)


def test_req_verify_6955_unknown_or_disagreement_is_quarantined(
    pairs: list[dict[str, object]],
) -> None:
    """REQ-VERIFY-6955 blocks any injected unknown or authority disagreement."""

    pair = pairs[0]
    enum = exp.prove_pair_with_enumerator(pair)
    unknown = exp.z3_failure_row(pair["pair_id"], "unknown")
    disagreement = deepcopy(unknown)
    disagreement["status"] = "proved"
    disagreement["label"] = "non_equivalent"

    unknown_agreement = exp.authority_agreement_row(pair, enum, unknown)
    bad_agreement = exp.authority_agreement_row(pair, enum, disagreement)

    assert unknown_agreement["quarantined"] is True
    assert unknown_agreement["reason"] == "z3_unknown"
    assert bad_agreement["quarantined"] is True
    assert bad_agreement["reason"] == "authority_disagreement"


def test_req_verify_6955_preconditions_and_blocked_artifact_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-6955 reports missing interfaces and unwritable checkpoint targets."""

    checkpoint = tmp_path / "checkpoint.json"
    checks = exp.check_preconditions(tmp_path, checkpoint)
    artifact = exp.build_blocked_artifact(
        date="20260903",
        repo_root=tmp_path,
        checkpoint_path=checkpoint,
        preconditions=checks,
        duration_s=0.001,
    )

    assert checks["current_constraint_interfaces"]["passed"] is False
    assert artifact["reformulation_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_reformulation_fixture"
    assert artifact["gate_check_summary"]["failed_check"] == "current_constraint_interfaces"
    assert artifact["gate_check_summary"]["expected"] is not None
    assert artifact["gate_check_summary"]["observed"] is not None


def test_req_verify_6955_run_writes_artifact_and_cli_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ready_artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6955 exposes the required artifact and command surfaces."""

    output = tmp_path / "artifact.json"
    checkpoint = tmp_path / "checkpoint.json"
    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: deepcopy(ready_artifact))
    artifact = exp.run(
        date="20260903",
        repo_root=ROOT,
        output_path=output,
        checkpoint_path=checkpoint,
    )
    summary_artifact = {
        "honest_verdict": artifact["honest_verdict"],
        "reformulation_fixture_ready_score": artifact["reformulation_fixture_ready_score"],
        "pair_rows": artifact["pair_rows"],
    }
    monkeypatch.setattr(exp, "run", lambda *, date: summary_artifact)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--date", "20260903"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "honest_verdict": artifact["honest_verdict"],
        "pair_count": 120,
        "reformulation_fixture_ready_score": 1,
    }


def test_req_verify_6955_formulation_schema_rejects_every_unsupported_shape(
    pairs: list[dict[str, object]],
) -> None:
    """REQ-VERIFY-6955 makes both engines consume one closed finite schema."""

    integer = deepcopy(pairs[0]["source"])
    boolean = deepcopy(
        next(pair for pair in pairs if pair["family"] == "boolean_cardinality")["source"]
    )
    piecewise = deepcopy(
        next(pair for pair in pairs if pair["family"] == "bounded_piecewise_linear")["source"]
    )
    cases: list[tuple[dict[str, object], str]] = []

    bad = deepcopy(integer)
    bad["extra"] = True
    cases.append((bad, "formulation_keys"))
    bad = deepcopy(integer)
    bad["schema_version"] = "wrong"
    cases.append((bad, "formulation_schema_version"))
    bad = deepcopy(integer)
    bad["variables"] = []
    cases.append((bad, "variables_nonempty"))
    bad = deepcopy(integer)
    bad["variables"][0] = None
    cases.append((bad, "variable_keys"))
    bad = deepcopy(integer)
    bad["variables"][0]["kind"] = "real"
    cases.append((bad, "variable_kind"))
    bad = deepcopy(integer)
    bad["variables"][0]["universe"] = []
    cases.append((bad, "finite_unique_universe"))
    bad = deepcopy(boolean)
    bad["variables"][0]["universe"] = [True, False]
    cases.append((bad, "boolean_universe"))
    bad = deepcopy(integer)
    bad["variables"][0]["universe"] = [True]
    cases.append((bad, "integer_universe"))
    bad = deepcopy(integer)
    bad["variables"][0]["domain"]["extra"] = 0
    cases.append((bad, "domain_keys"))
    bad = deepcopy(boolean)
    bad["variables"][0]["domain"]["lower"] = "0"
    cases.append((bad, "boolean_domain_bounds"))
    bad = deepcopy(integer)
    bad["variables"][1]["name"] = bad["variables"][0]["name"]
    cases.append((bad, "duplicate_formulation_variable"))
    bad = deepcopy(integer)
    bad["constraints"][0]["extra"] = 0
    cases.append((bad, "constraint_keys"))
    bad = deepcopy(integer)
    bad["constraints"][0]["op"] = "!="
    cases.append((bad, "constraint_operator"))
    bad = deepcopy(integer)
    bad["constraints"][0]["terms"] = {"missing": "1"}
    cases.append((bad, "constraint_variables"))
    bad = deepcopy(integer)
    bad["objective"]["extra"] = 0
    cases.append((bad, "objective_keys"))
    bad = deepcopy(integer)
    bad["objective"]["direction"] = "sideways"
    cases.append((bad, "objective_direction"))
    bad = deepcopy(piecewise)
    bad["objective"]["expression"]["pieces"][0]["extra"] = 0
    cases.append((bad, "linear_expression_keys"))
    bad = deepcopy(integer)
    bad["objective"]["expression"]["terms"] = {"missing": "1"}
    cases.append((bad, "objective_variables"))
    bad = deepcopy(integer)
    bad["objective"]["expression"] = None
    cases.append((bad, "objective_expression"))
    bad = deepcopy(integer)
    bad["objective"]["expression"]["extra"] = 0
    cases.append((bad, "linear_objective_keys"))
    bad = deepcopy(piecewise)
    bad["objective"]["expression"]["extra"] = 0
    cases.append((bad, "piecewise_objective_keys"))
    bad = deepcopy(piecewise)
    bad["objective"]["expression"]["aggregation"] = "median"
    cases.append((bad, "piecewise_aggregation"))
    bad = deepcopy(piecewise)
    bad["objective"]["expression"]["pieces"] = bad["objective"]["expression"]["pieces"][:1]
    cases.append((bad, "piecewise_pieces"))
    bad = deepcopy(integer)
    bad["objective"]["expression"]["kind"] = "quadratic"
    cases.append((bad, "objective_expression_kind"))

    for formulation, reason in cases:
        with pytest.raises(exp.FormulationSchemaError, match=reason):
            exp.validate_formulation(formulation)
    with pytest.raises(ValueError, match="non_exact_rational"):
        exp.as_fraction(0.5)
    with pytest.raises(ValueError, match="invalid_rational"):
        exp.as_fraction(object())


def test_scenario_verify_6955_mapping_rejects_all_closed_schema_boundaries(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-MAPPING covers every deterministic rejection code."""

    pair = next(row for row in pairs if row["expected_label"] == "equivalent")
    cases: list[tuple[dict[str, object], str]] = []

    bad = deepcopy(pair["mapping"])
    bad["schema_version"] = "wrong"
    cases.append((bad, "mapping_schema_version"))
    bad = deepcopy(pair["mapping"])
    bad["variables"] = {}
    cases.append((bad, "variable_mappings_list"))
    bad = deepcopy(pair["mapping"])
    bad["variables"][0] = None
    cases.append((bad, "variable_mapping_object"))
    bad = deepcopy(pair["mapping"])
    bad["variables"][0]["scale"] = "0"
    cases.append((bad, "zero_variable_scale"))
    bad = deepcopy(pair["mapping"])
    bad["variables"][1]["source"] = bad["variables"][0]["source"]
    cases.append((bad, "duplicate_source_variable"))
    bad = deepcopy(pair["mapping"])
    bad["variables"][0]["target"] = "not_declared"
    cases.append((bad, "target_variable_coverage"))
    bad = deepcopy(pair["mapping"])
    bad["domain_clauses"] = {}
    cases.append((bad, "domain_clauses_list"))
    bad = deepcopy(pair["mapping"])
    bad["domain_clauses"][0] = None
    cases.append((bad, "domain_clause_object"))
    bad = deepcopy(pair["mapping"])
    bad["domain_clauses"].pop()
    cases.append((bad, "domain_clause_coverage"))
    bad = deepcopy(pair["mapping"])
    bad["objective"] = None
    cases.append((bad, "objective_mapping_object"))
    bad = deepcopy(pair["mapping"])
    bad["objective"]["target_direction"] = "sideways"
    cases.append((bad, "mapping_objective_direction"))
    bad = deepcopy(pair["mapping"])
    bad["objective"]["scale"] = "0"
    cases.append((bad, "zero_objective_scale"))
    bad = deepcopy(pair["mapping"])
    bad["claimed_relation"] = "unknown"
    cases.append((bad, "claimed_relation"))

    for mapping, reason in cases:
        with pytest.raises(exp.MappingSchemaError, match=reason):
            exp.canonical_mapping(mapping, pair["source"], pair["target"])


def test_scenario_verify_6955_objective_order_counterexample_is_retained(
    pairs: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-6955-OBJECTIVE stores an order-only counterexample."""

    pair = deepcopy(
        next(
            row
            for row in pairs
            if row["expected_label"] == "equivalent"
            and exp.as_fraction(row["mapping"]["objective"]["scale"]) < 0
        )
    )
    pair["target"]["objective"]["direction"] = pair["source"]["objective"]["direction"]
    pair["mapping"]["objective"]["target_direction"] = pair["source"]["objective"]["direction"]
    pair["expected_label"] = "non_equivalent"
    pair["mapping"]["claimed_relation"] = "non_equivalent"
    result = exp.prove_pair_with_enumerator(pair)

    assert result["objective_affine_preserved"] is True
    assert result["objective_order_preserved"] is False
    assert result["counterexample"]["kind"] == "objective_order"


def test_req_verify_6955_timeout_and_expected_label_paths_fail_closed(
    pairs: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6955 quarantines solver uncertainty and frozen-label drift."""

    pair = pairs[0]
    enumeration = exp.prove_pair_with_enumerator(pair)
    monkeypatch.setattr(exp, "_solver_status", lambda *_clauses: ("unknown", "timeout simulated"))
    timeout = exp.prove_pair_with_z3(pair)
    monkeypatch.setattr(exp, "_solver_status", lambda *_clauses: ("unknown", "incomplete"))
    unknown = exp.prove_pair_with_z3(pair)
    wrong_pair = deepcopy(pair)
    wrong_pair["expected_label"] = "non_equivalent"
    expected_drift = exp.authority_agreement_row(
        wrong_pair,
        enumeration,
        {**exp.z3_failure_row(pair["pair_id"], "proved"), "label": "equivalent"},
    )
    monkeypatch.setattr(exp, "z3", None)
    unavailable = exp.prove_pair_with_z3(pair)

    assert timeout["status"] == "timeout"
    assert unknown["status"] == "unknown"
    assert expected_drift["reason"] == "expected_label_disagreement"
    assert unavailable["unknown_reasons"] == ["z3_unavailable"]


def test_req_verify_6955_solver_unknown_and_checkpoint_corruption_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6955 makes low-level unknown and corrupt replay inputs explicit."""

    class UnknownSolver:
        def set(self, **_kwargs: object) -> None:
            assert _kwargs

        def add(self, *_clauses: object) -> None:
            assert _clauses

        def check(self) -> str:
            return "indeterminate"

        def reason_unknown(self) -> str:
            return "test_unknown"

    monkeypatch.setattr(exp.z3, "Solver", UnknownSolver)
    assert exp._solver_status(exp.z3.BoolVal(True)) == ("unknown", "test_unknown")

    bad_schema = tmp_path / "bad-schema.json"
    bad_schema.write_text(json.dumps({"schema_version": "wrong", "pairs": []}), encoding="utf-8")
    bad_rows = tmp_path / "bad-rows.json"
    bad_rows.write_text(
        json.dumps({"schema_version": exp.CHECKPOINT_SCHEMA_VERSION, "pairs": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint_schema_version"):
        exp.replay_checkpoint(bad_schema)
    with pytest.raises(ValueError, match="checkpoint_pairs"):
        exp.replay_checkpoint(bad_rows)


def test_req_verify_6955_fresh_process_and_writable_failures_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6955 exposes child-process and checkpoint-write failures."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="child failed"),
    )
    with pytest.raises(RuntimeError, match="fresh_process_replay_failed"):
        exp._fresh_process_replay(ROOT, checkpoint)

    def write_non_rows(*_args: object, **_kwargs: object) -> SimpleNamespace:
        checkpoint.with_suffix(".replay.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exp.subprocess, "run", write_non_rows)
    with pytest.raises(RuntimeError, match="fresh_process_replay_not_rows"):
        exp._fresh_process_replay(ROOT, checkpoint)

    def unwritable(*_args: object, **_kwargs: object) -> object:
        raise OSError("read only")

    monkeypatch.setattr(exp.tempfile, "NamedTemporaryFile", unwritable)
    checks = exp.check_preconditions(ROOT, checkpoint)
    assert checks["writable_checkpoints"]["observed"]["error"] == "OSError"
    assert checks["all_passed"] is False


def test_req_verify_6955_build_block_and_artifact_validation_guards(
    tmp_path: Path,
    ready_artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6955 validates every score, verdict, count, and checksum boundary."""

    blocked = exp.build_artifact(
        date="20260903", repo_root=tmp_path, checkpoint_path=tmp_path / "corpus.json"
    )
    assert blocked["honest_verdict"] == "blocked_reformulation_fixture"

    variants: list[tuple[dict[str, object], str]] = []
    bad = deepcopy(ready_artifact)
    bad.pop("rows")
    variants.append((bad, "missing_required_fields"))
    bad = deepcopy(ready_artifact)
    bad["field_principles"].pop("rows")
    variants.append((bad, "missing_field_principles"))
    bad = deepcopy(ready_artifact)
    bad["reformulation_fixture_ready_score"] = 2
    variants.append((bad, "invalid_ready_score"))
    bad = deepcopy(ready_artifact)
    bad["gate_check_summary"]["passed"] = False
    variants.append((bad, "ready_gate_disagreement"))
    bad = deepcopy(ready_artifact)
    bad["pair_rows"].pop()
    variants.append((bad, "ready_pair_count"))
    bad = deepcopy(ready_artifact)
    bad["verdict_class"] = "positive"
    variants.append((bad, "ready_verdict_class"))
    bad = deepcopy(blocked)
    bad["verdict_class"] = "partial"
    variants.append((bad, "blocked_verdict_class"))
    bad = deepcopy(ready_artifact)
    bad["honest_verdict"] = "complete_wrong"
    variants.append((bad, "honest_verdict_not_circular_terminal"))
    bad = deepcopy(blocked)
    bad["honest_verdict"] = "complete_wrong"
    variants.append((bad, "honest_verdict_not_blocked_terminal"))
    bad = deepcopy(ready_artifact)
    bad["reproducibility_checksum"] = "sha256:wrong"
    variants.append((bad, "reproducibility_checksum_mismatch"))

    for artifact, reason in variants:
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(artifact)


def test_scenario_verify_6955_private_replay_cli_requires_and_writes_output(
    tmp_path: Path, pairs: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-6955-REPLAY covers the serialized child command surface."""

    checkpoint = tmp_path / "one.json"
    output = tmp_path / "one-replay.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": exp.CHECKPOINT_SCHEMA_VERSION,
                "random_seed": exp.RANDOM_SEED,
                "pairs": [pairs[0]],
            }
        ),
        encoding="utf-8",
    )

    assert exp.main(["--replay-checkpoint", str(checkpoint), "--replay-output", str(output)]) == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 1
    with pytest.raises(SystemExit):
        exp.main(["--replay-checkpoint", str(checkpoint)])
