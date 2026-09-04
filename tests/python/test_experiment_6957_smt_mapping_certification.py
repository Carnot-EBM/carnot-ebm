"""Tests for frozen SOTA mapping certification.

Spec refs: REQ-VERIFY-6957 and SCENARIO-VERIFY-6957-*.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from carnot import experiment_6955_reformulation_fixture as fixture_exp
from carnot import experiment_6957_smt_mapping_certification as exp
from scripts import adversarial_verify


ROOT = Path(__file__).resolve().parents[2]


def test_req_verify_6957_formal_substrate_is_not_live_model_inference(tmp_path: Path) -> None:
    """REQ-VERIFY-6957 treats frozen proposals plus two exact authorities as no-LLM work."""

    payload = {
        "experiment_id": "6957",
        "honest_verdict": "complete_null_sota_mapping_certification",
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "duration_s": 1.0,
        "random_seed": exp.RANDOM_SEED,
        "reproducibility_checksum": "sha256:fixture",
        "upstream_receipt": {"model_path": "frozen-proposals.gguf"},
    }
    path = tmp_path / "formal-certificate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    classification = adversarial_verify._classify_inference_substrate(payload)
    report = adversarial_verify.verify_artifact(path)
    flag_kinds = {row["kind"] for row in report["flags"]}

    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == exp.INFERENCE_SUBSTRATE
    assert "DURATION_TOO_SHORT" not in flag_kinds
    assert "METHODOLOGY_MISSING" not in flag_kinds


@pytest.fixture(scope="module")
def equivalent_pair() -> dict[str, object]:
    """Return one deterministic equivalent pair for isolated proof tests."""

    return next(
        pair
        for pair in fixture_exp.generate_pairs(fixture_exp.RANDOM_SEED)
        if pair["expected_label"] == "equivalent"
    )


def _attempt(
    pair: dict[str, object],
    *,
    mapping: dict[str, object] | None = None,
    json_valid: bool = True,
    schema_valid: bool = True,
    failure_reason: str | None = None,
    confidence: float | None = 0.9,
    rationale: str | None = "The declared affine map aligns both bounded domains.",
) -> dict[str, object]:
    """Build one frozen-attempt shape without invoking or repairing a parser."""

    candidate = None
    if json_valid:
        candidate = {
            "mapping": deepcopy(mapping if mapping is not None else pair["mapping"]),
            "confidence": confidence,
            "rationale": rationale,
        }
    return {
        "attempt_key": "model/example|pair|direct_affine",
        "hf_id": "model/example",
        "model_family": "example_model",
        "pair_id": pair["pair_id"],
        "problem_family": pair["family"],
        "prompt_variant_id": "direct_affine",
        "raw_sha256": "sha256:raw",
        "source_formulation": deepcopy(pair["source"]),
        "target_formulation": deepcopy(pair["target"]),
        "parse": {
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "failure_reason": failure_reason,
            "parsed_candidate": candidate,
            "confidence": confidence if json_valid else None,
            "rationale": rationale if json_valid else None,
        },
        "canonical_relation": pair["expected_label"],
        "difficulty": pair["difficulty"],
    }


def test_req_verify_6957_spec_precedes_code_and_declares_full_contract() -> None:
    """REQ-VERIFY-6957 owns every required field and named failure scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6957") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-VERIFY-6957-{name}" in section
        for name in (
            "PRECONDITIONS",
            "SCHEMA",
            "FEASIBILITY",
            "OBJECTIVE",
            "AUTHORITY",
            "METRICS",
            "SIGNALS",
            "REPLAY",
            "GATES",
        )
    )


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (lambda mapping: mapping["variables"].pop(), "source_variable_coverage"),
        (
            lambda mapping: mapping["variables"][1].update(
                {"target": mapping["variables"][0]["target"]}
            ),
            "duplicate_target_variable",
        ),
        (lambda mapping: mapping["variables"][0].update({"scale": "0"}), "zero_variable_scale"),
        (
            lambda mapping: mapping["variables"][0].update({"scale": "not-a-rational"}),
            "invalid_rational",
        ),
    ],
)
def test_scenario_verify_6957_schema_rejects_without_repair(
    equivalent_pair: dict[str, object], mutator: object, expected_reason: str
) -> None:
    """SCENARIO-VERIFY-6957-SCHEMA rejects missing, duplicate, and invalid affine data."""

    mapping = deepcopy(equivalent_pair["mapping"])
    mutator(mapping)
    attempt = _attempt(equivalent_pair, mapping=mapping)
    result = exp.certify_proposal(attempt)

    assert result["schema_row"]["schema_valid"] is False
    assert expected_reason in result["schema_row"]["failure_reason"]
    assert result["variable_coverage_row"]["coverage_complete"] is False or expected_reason in {
        "zero_variable_scale",
        "invalid_rational",
    }
    assert result["z3_row"]["status"] == "schema_rejected"
    assert result["enumeration_row"]["status"] == "schema_rejected"
    assert result["authority_agreement_row"]["authorities_agree"] is True
    assert result["proposal_row"]["exact_mapping_correct"] is False
    assert attempt["parse"]["parsed_candidate"]["mapping"] == mapping


def test_scenario_verify_6957_schema_keeps_malformed_parse_in_denominator(
    equivalent_pair: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-SCHEMA gives malformed bytes two terminal parse rejections."""

    result = exp.certify_proposal(
        _attempt(
            equivalent_pair,
            json_valid=False,
            schema_valid=False,
            failure_reason="malformed_json",
        )
    )

    assert result["parse_row"]["parse_failure"] is True
    assert result["z3_row"]["status"] == "parse_rejected"
    assert result["enumeration_row"]["status"] == "parse_rejected"
    assert result["authority_agreement_row"]["terminal"] is True
    assert result["proposal_row"]["certified_relation"] is None


def test_scenario_verify_6957_feasibility_checks_forward_and_reverse(
    equivalent_pair: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-FEASIBILITY preserves cross-domain counterexamples."""

    mapping = deepcopy(equivalent_pair["mapping"])
    mapping["variables"][0]["offset"] = "99"
    result = exp.certify_proposal(_attempt(equivalent_pair, mapping=mapping))

    enum = result["enumeration_row"]
    z3_row = result["z3_row"]
    assert enum["status"] == "counterexample"
    assert z3_row["status"] == "counterexample"
    assert not enum["forward_feasible"] or not enum["reverse_feasible"]
    assert not z3_row["forward_feasible"] or not z3_row["reverse_feasible"]
    assert result["cross_feasibility_row"]["authorities_agree"] is True
    assert any(
        row["kind"] in {"forward_feasibility", "reverse_feasibility"}
        for row in result["counterexample_rows"]
    )


def test_scenario_verify_6957_feasibility_rejects_fractional_integer_map(
    equivalent_pair: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-FEASIBILITY does not truncate a fractional target value."""

    mapping = deepcopy(equivalent_pair["mapping"])
    mapping["variables"][0]["scale"] = "1/2"
    result = exp.certify_proposal(_attempt(equivalent_pair, mapping=mapping))

    assert result["enumeration_row"]["forward_feasible"] is False
    assert result["z3_row"]["forward_feasible"] is False
    witness = result["enumeration_row"]["counterexamples"]["forward_feasibility"]
    assert "/" in json.dumps(witness)


def test_scenario_verify_6957_objective_reversal_affine_and_ties(
    equivalent_pair: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-OBJECTIVE checks direction, affine values, global order, and ties."""

    wrong_direction = deepcopy(equivalent_pair["mapping"])
    scale = fixture_exp.as_fraction(wrong_direction["objective"]["scale"])
    wrong_direction["objective"]["target_direction"] = wrong_direction["objective"][
        "source_direction"
    ]
    if scale > 0:
        wrong_direction["objective"]["target_direction"] = (
            "max" if wrong_direction["objective"]["source_direction"] == "min" else "min"
        )
    direction_result = exp.certify_proposal(_attempt(equivalent_pair, mapping=wrong_direction))
    assert direction_result["objective_direction_row"]["direction_valid"] is False
    assert direction_result["proposal_row"]["certified_relation"] == "non_equivalent"

    wrong_affine = deepcopy(equivalent_pair["mapping"])
    wrong_affine["objective"]["offset"] = "999"
    affine_result = exp.certify_proposal(_attempt(equivalent_pair, mapping=wrong_affine))
    assert affine_result["enumeration_row"]["objective_affine_preserved"] is False
    assert affine_result["z3_row"]["objective_affine_preserved"] is False

    tied_pair = deepcopy(equivalent_pair)
    for side in ("source", "target"):
        objective = tied_pair[side]["objective"]
        objective["expression"] = {
            "kind": "linear",
            "terms": {row["name"]: "0" for row in tied_pair[side]["variables"]},
            "constant": "0",
        }
    tied_pair["mapping"]["objective"].update(
        {
            "source_direction": tied_pair["source"]["objective"]["direction"],
            "target_direction": tied_pair["target"]["objective"]["direction"],
            "scale": "1",
            "offset": "0",
        }
    )
    tie_result = exp.certify_proposal(_attempt(tied_pair))
    assert tie_result["enumeration_row"]["objective_order_preserved"] is True
    assert tie_result["z3_row"]["objective_order_preserved"] is True
    assert tie_result["objective_order_row"]["tie_count"] > 0


@pytest.mark.parametrize("status", ["timeout", "unknown"])
def test_scenario_verify_6957_authority_quarantines_z3_nondecisions(
    equivalent_pair: dict[str, object], status: str
) -> None:
    """SCENARIO-VERIFY-6957-AUTHORITY retains timeout and unknown rows as terminal."""

    def failed_z3(pair: dict[str, object]) -> dict[str, object]:
        return exp.engine_failure_row(pair["pair_id"], "z3", status, status)

    result = exp.certify_proposal(_attempt(equivalent_pair), z3_certifier=failed_z3)

    assert result["z3_row"]["status"] == status
    assert result["authority_agreement_row"]["quarantined"] is True
    assert result["authority_agreement_row"]["terminal"] is True
    assert result["proposal_row"][status] is True
    assert result["proposal_row"]["exact_mapping_correct"] is False


def test_scenario_verify_6957_authority_quarantines_enumerator_disagreement(
    equivalent_pair: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-AUTHORITY catches a second-engine label disagreement."""

    def disagreeing_enumerator(pair: dict[str, object]) -> dict[str, object]:
        row = exp.certify_with_enumerator(pair)
        row["label"] = "non_equivalent"
        row["status"] = "counterexample"
        row["objective_order_preserved"] = False
        return row

    result = exp.certify_proposal(
        _attempt(equivalent_pair), enumeration_certifier=disagreeing_enumerator
    )

    agreement = result["authority_agreement_row"]
    assert agreement["authorities_agree"] is False
    assert agreement["quarantined"] is True
    assert agreement["reason"] == "authority_disagreement"


def test_scenario_verify_6957_metrics_counts_false_acceptance_and_rejection() -> None:
    """SCENARIO-VERIFY-6957-METRICS reduces every outcome from proposal rows."""

    rows = [
        exp.synthetic_metric_row("a", "equivalent", "equivalent", parse_failure=False),
        exp.synthetic_metric_row("b", "non_equivalent", "equivalent", parse_failure=False),
        exp.synthetic_metric_row("c", "equivalent", "non_equivalent", parse_failure=False),
        exp.synthetic_metric_row("d", "equivalent", None, parse_failure=True),
    ]
    metric = exp.metric_row("overall", "all", rows)

    assert metric["proposal_count"] == 4
    assert metric["exact_mapping_correct_count"] == 1
    assert metric["exact_mapping_accuracy"] == 0.25
    assert metric["false_acceptance_count"] == 1
    assert metric["false_rejection_count"] == 1
    assert metric["parse_failure_count"] == 1
    assert metric["unclassified_count"] == 1


def test_scenario_verify_6957_signals_are_tie_aware_and_advisory() -> None:
    """SCENARIO-VERIFY-6957-SIGNALS assigns half credit to ties and null to one class."""

    assert exp.tie_aware_auroc([(0.8, True), (0.8, False)]) == 0.5
    assert exp.tie_aware_auroc([(0.8, True)]) is None
    rows = [
        {"confidence": 0.95, "exact_mapping_correct": True, "rationale_present": True},
        {"confidence": 0.95, "exact_mapping_correct": False, "rationale_present": False},
        {"confidence": None, "exact_mapping_correct": False, "rationale_present": True},
    ]
    calibration = exp.calibration_row("overall", "all", rows)

    assert calibration["confidence_count"] == 2
    assert calibration["confidence_auroc"] == 0.5
    assert calibration["top_confidence_error_rate"] == 0.5
    assert calibration["rationale_present_count"] == 2
    assert calibration["rationale_absent_count"] == 1


def test_scenario_verify_6957_gates_require_strict_ci_and_low_false_acceptance() -> None:
    """SCENARIO-VERIFY-6957-GATES does not promote a zero-bound tie."""

    tied = [
        {"model_family": "a", "ci95_lower": 0.0},
        {"model_family": "b", "ci95_lower": 0.1},
        {"model_family": "c", "ci95_lower": 0.2},
    ]
    assert exp.reduce_positive_score(1, tied, 0.0) == 1
    tied[2]["ci95_lower"] = 0.0
    assert exp.reduce_positive_score(1, tied, 0.0) == 0
    tied[2]["ci95_lower"] = 0.2
    assert exp.reduce_positive_score(1, tied, 0.01) == 0
    assert exp.reduce_positive_score(0, tied, 0.0) == 0

    interval = exp.paired_bootstrap_interval([0, 0, 0], seed=7, resamples=25)
    assert interval == {"mean_delta": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
    assert exp.paired_bootstrap_interval([], seed=7, resamples=25) == {
        "mean_delta": None,
        "ci95_lower": None,
        "ci95_upper": None,
    }


def test_req_verify_6957_exact_helper_boundaries(
    equivalent_pair: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6957 covers typed domains, nondecisions, and evidence boundaries."""

    malformed_coverage = exp._variable_coverage(
        "key", equivalent_pair["source"], equivalent_pair["target"], {"variables": {}}
    )
    assert malformed_coverage["coverage_complete"] is False

    integer_formulation = equivalent_pair["source"]
    outside = {
        row["name"]: Fraction(max(row["universe"]) + 1) for row in integer_formulation["variables"]
    }
    assert exp._coerce_assignment(integer_formulation, outside) is None
    boolean_pair = next(
        pair
        for pair in fixture_exp.generate_pairs(fixture_exp.RANDOM_SEED)
        if pair["family"] == "boolean_cardinality" and pair["expected_label"] == "equivalent"
    )
    invalid_boolean = {row["name"]: Fraction(2) for row in boolean_pair["source"]["variables"]}
    assert exp._coerce_assignment(boolean_pair["source"], invalid_boolean) is None

    pair = {
        "pair_id": "unknown",
        "source": equivalent_pair["source"],
        "target": equivalent_pair["target"],
        "mapping": equivalent_pair["mapping"],
    }
    real_query = exp._z3_query
    monkeypatch.setattr(exp, "_z3_query", lambda *args, **kwargs: ("unknown", None, "reason"))
    unknown = exp.certify_with_z3(pair)
    assert unknown["status"] == "unknown"
    assert unknown["unknown_reasons"] == ["reason"] * 4
    monkeypatch.setattr(exp, "_z3_query", real_query)

    class UnknownSolver:
        """Return a real Z3-style nondecision without running an unstable timeout."""

        reason = "timeout"

        def set(self, **kwargs: object) -> None:
            assert kwargs

        def add(self, *args: object) -> None:
            assert args

        def check(self) -> str:
            return "indeterminate"

        def reason_unknown(self) -> str:
            return self.reason

    monkeypatch.setattr(fixture_exp.z3, "Solver", UnknownSolver)
    assert exp._z3_query(True)[0] == "timeout"
    UnknownSolver.reason = "incomplete"
    assert exp._z3_query(True)[0] == "unknown"

    enum_unknown = exp.engine_failure_row("key", "enumerator", "unknown", "reason")
    z3_proved = exp.certify_with_enumerator(pair)
    agreement = exp.authority_agreement_row("key", enum_unknown, z3_proved)
    assert agreement["reason"] == "enumerator_unknown"
    assert exp._engine_evidence_rows("key", {"engine": "x", "bad": []}, "bad") == []


def test_req_verify_6957_input_validation_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-6957 fails closed for malformed rosters, bindings, and replay files."""

    assert exp._raw_roster_valid({"attempt_rows": {}})["hashes_match"] is False
    broken_bank = {
        "attempt_rows": [None, {"attempt_key": "missing"}, {"attempt_key": "bad", "raw_text": "x"}],
        "raw_output_rows": [{"attempt_key": "bad", "raw_text": "different", "raw_sha256": "wrong"}],
    }
    assert exp._raw_roster_valid(broken_bank)["hashes_match"] is False
    binding = exp._fixture_binding_valid(
        {"attempt_rows": [None, {"pair_id": "missing"}]}, {}, {"pairs": []}
    )
    assert binding["matched_attempt_count"] == 0

    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"schema_version": "wrong", "inputs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_schema_version"):
        exp.replay_checkpoint(replay)
    replay.write_text(
        json.dumps({"schema_version": exp.REPLAY_SCHEMA_VERSION, "inputs": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="replay_inputs"):
        exp.replay_checkpoint(replay)

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_json(array_path)

    blocked = exp.build_from_paths(
        run_date="20260903",
        repo_root=tmp_path,
        bank_path=tmp_path / "missing-bank.json",
        fixture_path=tmp_path / "missing-fixture.json",
        fixture_checkpoint_path=tmp_path / "missing-checkpoint.json",
        replay_checkpoint_path=tmp_path / "replay-checkpoint.json",
    )
    assert blocked["gate_check_summary"][0]["failed_check"] == "frozen_input_readable"

    for name in ("bank", "fixture", "checkpoint"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
    blocked = exp.build_from_paths(
        run_date="20260903",
        repo_root=tmp_path,
        bank_path=tmp_path / "bank.json",
        fixture_path=tmp_path / "fixture.json",
        fixture_checkpoint_path=tmp_path / "checkpoint.json",
        replay_checkpoint_path=tmp_path / "replay-checkpoint.json",
    )
    assert blocked["verdict_class"] == "blocked"
    assert any(
        row["failed_check"] == "reformulation_bank_complete_score"
        for row in blocked["gate_check_summary"]
    )
    assert exp._outcome(0, 0)[0] == "partial"
    assert exp._outcome(1, 1)[0] == "positive"
    assert exp._outcome(1, 0)[0] == "null"


def test_scenario_verify_6957_preconditions_write_complete_blocked_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6957-PRECONDITIONS records exact expected and observed failures."""

    preconditions = [exp.gate_check("bank_complete", 1, 0)]
    artifact = exp.build_blocked_artifact(
        run_date="20260903",
        duration_s=0.01,
        preconditions_checked=preconditions,
        source_artifact_hashes={},
    )

    assert artifact["honest_verdict"] == "blocked_smt_mapping_certification"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["smt_certification_run_complete_score"] == 0
    assert artifact["sota_mapping_positive_score"] == 0
    assert artifact["gate_check_summary"] == [
        {"failed_check": "bank_complete", "expected_value": 1, "observed_value": 0}
    ]
    assert all(field in artifact for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(artifact) == []

    output = tmp_path / "blocked.json"
    exp.write_json_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the complete real frozen bank once against a temporary replay checkpoint."""

    directory = tmp_path_factory.mktemp("exp6957")
    return exp.build_from_paths(
        run_date="20260903",
        repo_root=ROOT,
        bank_path=ROOT / exp.BANK_PATH,
        fixture_path=ROOT / exp.FIXTURE_PATH,
        fixture_checkpoint_path=ROOT / exp.FIXTURE_CHECKPOINT_PATH,
        replay_checkpoint_path=directory / "checkpoint.json",
    )


def test_scenario_verify_6957_replay_and_complete_artifact(
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6957-REPLAY recomputes all certificates and headline metrics."""

    artifact = complete_artifact
    assert len(artifact["proposal_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["z3_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["enumeration_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["authority_agreement_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["fresh_process_replay_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert all(row["replay_matches"] for row in artifact["fresh_process_replay_rows"])
    assert all(row["headline_metrics_match"] for row in artifact["fresh_process_replay_rows"])
    assert artifact["smt_certification_run_complete_score"] == 1
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert not exp.validate_artifact(artifact)

    assert len(artifact["model_rows"]) == 3
    assert len(artifact["family_rows"]) == 3
    assert {row["difficulty"] for row in artifact["difficulty_rows"]} == {
        "standard",
        "hard",
    }
    assert len(artifact["prompt_variant_rows"]) == 3
    assert all(row["proposal_count"] > 0 for row in artifact["model_rows"])
    assert len(artifact["baseline_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["paired_metric_rows"]) == exp.EXPECTED_PROPOSAL_COUNT
    assert len(artifact["confidence_interval_rows"]) == 3


def test_req_verify_6957_validation_and_private_replay_cli(
    tmp_path: Path, complete_artifact: dict[str, object]
) -> None:
    """REQ-VERIFY-6957 rejects drift and exposes the serialized replay command."""

    bad = deepcopy(complete_artifact)
    bad["proposal_rows"].pop()
    assert "proposal_row_count" in exp.validate_artifact(bad)
    bad = deepcopy(complete_artifact)
    bad["field_principles"].pop("rows")
    assert "field_principles_incomplete" in exp.validate_artifact(bad)
    bad = deepcopy(complete_artifact)
    bad["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad)

    variants = []
    bad = deepcopy(complete_artifact)
    bad.pop("rows")
    variants.append((bad, "missing_required_fields"))
    bad = deepcopy(complete_artifact)
    bad["inference_substrate"] = "wrong"
    variants.append((bad, "inference_substrate_mismatch"))
    bad = deepcopy(complete_artifact)
    bad["verifier_is_oracle"] = True
    variants.append((bad, "verifier_is_oracle_must_be_false"))
    bad = deepcopy(complete_artifact)
    bad["smt_certification_run_complete_score"] = 0
    variants.append((bad, "completion_score_mismatch"))
    bad = deepcopy(complete_artifact)
    bad["sota_mapping_positive_score"] = 1 - bad["sota_mapping_positive_score"]
    variants.append((bad, "positive_score_mismatch"))
    for artifact, error in variants:
        assert any(value.startswith(error) for value in exp.validate_artifact(artifact))

    blocked = exp.build_blocked_artifact(
        run_date="20260903",
        duration_s=0.1,
        preconditions_checked=[exp.gate_check("x", 1, 0)],
        source_artifact_hashes={},
    )
    blocked["smt_certification_run_complete_score"] = 1
    assert "blocked_scores_nonzero" in exp.validate_artifact(blocked)
    for verdict_class, verdict in (
        ("blocked", "wrong"),
        ("partial", "wrong"),
        ("null", "wrong"),
    ):
        bad = deepcopy(complete_artifact if verdict_class != "blocked" else blocked)
        bad["verdict_class"] = verdict_class
        bad["honest_verdict"] = verdict
        assert "honest_verdict_prefix_mismatch" in exp.validate_artifact(bad)
    bad = deepcopy(complete_artifact)
    bad["sota_mapping_positive_score"] = 1
    bad["verdict_class"] = "null"
    assert "positive_verdict_class_mismatch" in exp.validate_artifact(bad)
    bad = deepcopy(complete_artifact)
    bad["sota_mapping_positive_score"] = 0
    bad["verdict_class"] = "positive"
    assert "null_verdict_class_mismatch" in exp.validate_artifact(bad)

    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "replay.json"
    one_input = exp.serialize_attempt_input(_attempt(exp.deserialize_fixture_pair(equivalent=True)))
    exp.write_json_atomic(
        checkpoint,
        {
            "schema_version": exp.REPLAY_SCHEMA_VERSION,
            "random_seed": exp.RANDOM_SEED,
            "inputs": [one_input],
        },
    )
    assert (
        exp.main(
            [
                "--replay-checkpoint",
                str(checkpoint),
                "--replay-output",
                str(output),
            ]
        )
        == 0
    )
    replay = json.loads(output.read_text(encoding="utf-8"))
    assert len(replay["proposal_rows"]) == 1
    with pytest.raises(SystemExit):
        exp.main(["--replay-checkpoint", str(checkpoint)])


def test_req_verify_6957_run_and_normal_cli_write_validated_output(
    tmp_path: Path,
    complete_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6957 covers the validated writer and normal command summary."""

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(exp, "build_from_paths", lambda **kwargs: deepcopy(complete_artifact))
    written = exp.run(date="20260903", repo_root=tmp_path, output_path=output)
    assert written["experiment_id"] == 6957
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 6957

    invalid = deepcopy(complete_artifact)
    invalid["reproducibility_checksum"] = "bad"
    monkeypatch.setattr(exp, "build_from_paths", lambda **kwargs: invalid)
    with pytest.raises(ValueError, match="artifact_validation"):
        exp.run(date="20260903", repo_root=tmp_path, output_path=output)

    monkeypatch.setattr(exp, "run", lambda **kwargs: deepcopy(complete_artifact))
    assert exp.main(["--date", "20260903"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["proposal_count"] == exp.EXPECTED_PROPOSAL_COUNT
