"""Focused tests for REQ-VERIFY-7197 and SCENARIO-VERIFY-7197-*.

The tests use frozen upstream rows. All generated files stay in private test
directories, so the research record cannot change during validation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7197_v634_grounding_value_audit as exp
from carnot import experiment_7197_v634_grounding_value_independent_audit as auditor


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
PUBLIC = REPO / "results/experiment_7195_v634_typed_grounding_public.jsonl"
AUTHORITY = REPO / "results/experiment_7195_v634_typed_grounding_authority.jsonl"
CAPTURE = REPO / "results/experiment_7196_v634_qwen_atomic_capture.json"


def _jsonl(path: Path) -> list[dict[str, object]]:
    """Load immutable line records for one test fixture."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def sources() -> dict[str, object]:
    """Load the sealed panel and cached calls once for focused tests."""

    return {
        "public": _jsonl(PUBLIC),
        "authority": _jsonl(AUTHORITY),
        "capture": json.loads(CAPTURE.read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def features(sources: dict[str, object]) -> list[dict[str, object]]:
    """Reparse model bytes without supplying evaluator labels."""

    return exp.build_candidate_features(sources["public"], sources["capture"]["completion_rows"])


@pytest.fixture(scope="module")
def predictions(
    sources: dict[str, object], features: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Freeze all five policies before evaluator labels are scored."""

    donor_map = exp.build_shuffle_map(sources["authority"])
    return exp.evaluate_arms(features, donor_map)


@pytest.fixture(scope="module")
def scored(
    sources: dict[str, object], predictions: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Open held-out outcomes only after predictions are complete."""

    return exp.score_evaluation(predictions, sources["authority"])


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], dict[str, Path]]:
    """Build one full artifact through the fresh evaluator process."""

    directory = tmp_path_factory.mktemp("exp7197")
    paths = {
        "result": directory / "result.json",
        "checkpoint": directory / "checkpoints" / "running.json",
        "audit_request": directory / "checkpoints" / "audit_request.json",
        "audit_result": directory / "checkpoints" / "audit_result.json",
    }
    paths["checkpoint"].parent.mkdir()
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=paths["result"],
        checkpoint_path=paths["checkpoint"],
        audit_request_path=paths["audit_request"],
        audit_result_path=paths["audit_result"],
        duration_s=0.25,
    )
    return artifact, paths


def _completion(
    unit_id: str,
    call_type: str,
    value: object,
    *,
    latency_s: float = 0.1,
) -> dict[str, object]:
    """Create one exact raw-call receipt for a small semantic fixture."""

    raw = exp.canonical_json(value)
    return {
        "unit_id": unit_id,
        "call_id": f"000:{call_type}",
        "call_type": call_type,
        "row_order": 0,
        "raw_output": raw,
        "raw_output_sha256": exp.sha256_text(raw),
        "latency_s": latency_s,
        "cache_hit": False,
        "cold_request": True,
        "reuse_from_call_id": None,
        "parse_status": "producer_projection_is_not_trusted",
        "parsed_output": {"wrong": True},
        "request_error": None,
        "truncated": False,
    }


def _semantic_fixture(*, reverse_claim: bool = False) -> tuple[list[dict], list[dict]]:
    """Make valid separated source and claim outputs with different entity IDs."""

    unit_id = "unit-small"
    source = "entity-a precedes entity-b."
    claim = "entity-b precedes entity-a." if reverse_claim else source
    public = [{"unit_id": unit_id, "source_text": source, "claim_text": claim}]
    source_value = {
        "entity_bindings": [
            {"entity_id": "source-a", "surface": "entity-a", "source_start": 0, "source_end": 8},
            {
                "entity_id": "source-b",
                "surface": "entity-b",
                "source_start": 18,
                "source_end": 26,
            },
        ],
        "relations": [
            {
                "subject_id": "source-a",
                "operator": "precedes",
                "object_id": "source-b",
                "polarity": "positive",
                "source_start": 0,
                "source_end": len(source),
            }
        ],
        "missing_fields": [],
    }
    claim_value = {
        "entity_bindings": [
            {
                "entity_id": "claim-a",
                "surface": "entity-a",
                "source_start": 18 if reverse_claim else 0,
                "source_end": 26 if reverse_claim else 8,
            },
            {
                "entity_id": "claim-b",
                "surface": "entity-b",
                "source_start": 0 if reverse_claim else 18,
                "source_end": 8 if reverse_claim else 26,
            },
        ],
        "relations": [
            {
                "subject_id": "claim-b" if reverse_claim else "claim-a",
                "operator": "precedes",
                "object_id": "claim-a" if reverse_claim else "claim-b",
                "polarity": "positive",
                "source_start": 0,
                "source_end": len(claim),
            }
        ],
        "missing_fields": [],
    }
    calls = [
        _completion(unit_id, "source", source_value),
        _completion(unit_id, "claim", claim_value),
        _completion(unit_id, "direct", {"decision": "supported"}),
    ]
    return public, calls


def test_req_verify_7197_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7197 defines all focused scenarios and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7197") :]
    for scenario in (
        "PREFLIGHT",
        "DENOMINATOR",
        "POLICIES",
        "METRICS",
        "AUDIT",
        "VALUE",
        "CIRCULARITY",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7197-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7197_policies_reparse_raw_bytes(
    sources: dict[str, object], features: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-7197-POLICIES ignores producer parse projections."""

    assert len(features) == 192
    changed = deepcopy(sources["capture"]["completion_rows"])
    changed[0]["parsed_output"] = {"invented": "authority"}
    changed[0]["parse_status"] = "valid"
    assert exp.build_candidate_features(sources["public"], changed) == features

    changed[0]["raw_output"] = "{}"
    with pytest.raises(ValueError, match="raw_output_hash_mismatch"):
        exp.build_candidate_features(sources["public"], changed)


def test_scenario_verify_7197_policies_are_label_invariant(
    sources: dict[str, object],
    features: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-POLICIES freezes decisions without labels."""

    changed = deepcopy(sources["authority"])
    for row in changed:
        row["support_label"] = "unsupported" if row["support_label"] == "supported" else "supported"
        row["expected_executor_decision"] = "changed"
    assert exp.build_shuffle_map(changed) == exp.build_shuffle_map(sources["authority"])
    assert exp.prediction_checksum(predictions) == exp.prediction_checksum(
        exp.evaluate_arms(features, exp.build_shuffle_map(changed))
    )
    encoded = exp.canonical_json(predictions)
    assert "support_label" not in encoded
    assert "expected_executor_decision" not in encoded


def test_scenario_verify_7197_policies_distinguish_syntax_and_semantics() -> None:
    """SCENARIO-VERIFY-7197-DENOMINATOR does not treat valid JSON as truth."""

    public, calls = _semantic_fixture(reverse_claim=True)
    features = exp.build_candidate_features(public, calls)
    rows = exp.evaluate_arms(features, {"unit-small": "unit-small"})
    by_arm = {row["arm"]: row for row in rows}
    assert by_arm["grammar_validity_only"]["prediction"] == "supported"
    assert by_arm["typed_execution_unknown_abstention"]["prediction"] == "unsupported"
    assert by_arm["direct_judgment"]["prediction"] == "supported"
    # Token-set overlap cannot see argument order. This is why it is a
    # semantics-free control rather than a correctness mechanism.
    assert by_arm["lexical_overlap"]["prediction"] == "supported"


def test_scenario_verify_7197_denominator_unknown_abstains() -> None:
    """SCENARIO-VERIFY-7197-DENOMINATOR keeps failed extraction as unknown."""

    public, calls = _semantic_fixture()
    calls[1]["raw_output"] = "not-json"
    calls[1]["raw_output_sha256"] = exp.sha256_text("not-json")
    rows = exp.evaluate_arms(
        exp.build_candidate_features(public, calls), {"unit-small": "unit-small"}
    )
    by_arm = {row["arm"]: row for row in rows}
    for arm in (
        "grammar_validity_only",
        "typed_execution_unknown_abstention",
        "shuffled_source_typed_execution",
    ):
        assert by_arm[arm]["prediction"] == "abstain"
        assert by_arm[arm]["abstention"] is True
        assert by_arm[arm]["error"]


def test_scenario_verify_7197_denominator_keeps_all_real_rows(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-DENOMINATOR retains five copies of 128 units."""

    assert len(scored) == 5 * 128
    assert exp.evaluation_row_errors(scored) == []
    metrics = exp.summarize_arms(scored)
    assert set(metrics) == set(exp.ARMS)
    assert all(metric["denominator"] == 128 for metric in metrics.values())
    typed = metrics["typed_execution_unknown_abstention"]
    assert typed["parse_success_count"] == 0
    assert typed["coverage_count"] == 0
    assert typed["accuracy_count"] == 0
    assert typed["conditional_accuracy"] is None
    assert typed["abstention_count"] == 128


def test_scenario_verify_7197_metrics_reject_changed_pair_rosters(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-METRICS detects drops, duplicates, and pair drift."""

    assert "row_count" in exp.evaluation_row_errors(scored[:-1])
    duplicate = deepcopy(scored)
    duplicate[-1]["unit_id"] = duplicate[-2]["unit_id"]
    assert any("unit_roster" in error for error in exp.evaluation_row_errors(duplicate))
    changed = deepcopy(scored)
    changed[128]["base_id"] = "wrong-base"
    assert any("pair_roster" in error for error in exp.evaluation_row_errors(changed))


def test_scenario_verify_7197_metrics_report_required_rates(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-METRICS reports quality, coverage, edits, and cost."""

    metrics = exp.summarize_arms(scored)
    required = {
        "accuracy_count",
        "accuracy",
        "parse_rate",
        "abstention_rate",
        "coverage",
        "conditional_accuracy",
        "false_accept_count",
        "false_reject_count",
        "harmful_flip_count",
        "rename_consistency",
        "semantic_edit_sensitivity",
        "cold_latency_s",
        "amortized_latency_s",
    }
    assert all(required <= set(metric) for metric in metrics.values())
    assert metrics["direct_judgment"]["coverage"] == 1.0
    assert metrics["lexical_overlap"]["accuracy"] == 1.0
    assert metrics["typed_execution_unknown_abstention"]["harmful_flip_count"] == 0


def test_scenario_verify_7197_metrics_bootstrap_clusters_base_cases(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-METRICS uses fixed family-stratified base draws."""

    first = exp.paired_cluster_bootstrap(scored, exp.BOOTSTRAP_SEED, draws=10_000)
    second = exp.paired_cluster_bootstrap(scored, exp.BOOTSTRAP_SEED, draws=10_000)
    assert first == second
    assert len(first) >= 3
    assert all(row["draw_count"] == 10_000 for row in first)
    assert all(row["cluster_count"] == 32 for row in first)
    assert all(row["stratified_by"] == "relation_family" for row in first)
    assert all(row["independent_unit"] == "base_id" for row in first)


def test_scenario_verify_7197_value_accuracy_branch_is_strict() -> None:
    """SCENARIO-VERIFY-7197-VALUE requires gain, safety, coverage, and audit."""

    metrics = {
        "direct_judgment": {"coverage": 1.0, "amortized_latency_s": 2.0},
        "typed_execution_unknown_abstention": {
            "coverage": 0.75,
            "amortized_latency_s": 2.0,
        },
    }
    intervals = [
        {
            "comparison": "typed_vs_direct",
            "metric": "accuracy",
            "estimate": 0.1,
            "ci95_lower": 0.01,
            "ci95_upper": 0.2,
        },
        {
            "comparison": "typed_vs_direct",
            "metric": "false_accept_rate",
            "estimate": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
        },
        {
            "comparison": "typed_vs_direct",
            "metric": "coverage",
            "estimate": -0.25,
            "ci95_lower": -0.3,
            "ci95_upper": -0.2,
        },
    ]
    verdict = exp.classify_value(metrics, intervals, independent_audit_passed=True)
    assert verdict["accuracy_criterion"]["passed"] is True
    assert verdict["efficiency_criterion"]["passed"] is False
    assert verdict["acceptance_gate_value"] == 1
    assert verdict["grounding_value_score"] == 1
    assert verdict["verdict_class"] == "circular_positive"
    assert verdict["verifier_is_oracle"] is True

    intervals[0]["ci95_lower"] = 0.0
    assert (
        exp.classify_value(metrics, intervals, independent_audit_passed=True)["accuracy_criterion"][
            "passed"
        ]
        is False
    )


def test_scenario_verify_7197_value_efficiency_is_an_alternative() -> None:
    """SCENARIO-VERIFY-7197-VALUE accepts noninferior equal-coverage speed."""

    metrics = {
        "direct_judgment": {"coverage": 1.0, "amortized_latency_s": 4.0},
        "typed_execution_unknown_abstention": {
            "coverage": 1.0,
            "amortized_latency_s": 1.5,
        },
    }
    intervals = [
        {
            "comparison": "typed_vs_direct",
            "metric": "accuracy",
            "estimate": -0.01,
            "ci95_lower": -0.02,
            "ci95_upper": 0.01,
        },
        {
            "comparison": "typed_vs_direct",
            "metric": "false_accept_rate",
            "estimate": 0.1,
            "ci95_lower": 0.0,
            "ci95_upper": 0.2,
        },
        {
            "comparison": "typed_vs_direct",
            "metric": "coverage",
            "estimate": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
        },
    ]
    verdict = exp.classify_value(metrics, intervals, independent_audit_passed=True)
    assert verdict["accuracy_criterion"]["passed"] is False
    assert verdict["efficiency_criterion"]["passed"] is True
    assert verdict["acceptance_gate_value"] == 1

    metrics["typed_execution_unknown_abstention"]["coverage"] = 0.99
    assert (
        exp.classify_value(metrics, intervals, independent_audit_passed=True)[
            "efficiency_criterion"
        ]["passed"]
        is False
    )


def test_scenario_verify_7197_value_complete_null_preserves_completion(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-VALUE keeps a low-coverage audit complete."""

    metrics = exp.summarize_arms(scored)
    intervals = exp.paired_cluster_bootstrap(scored, exp.BOOTSTRAP_SEED, draws=200)
    verdict = exp.classify_value(metrics, intervals, independent_audit_passed=True)
    assert verdict["grounding_audit_complete_score"] == 1
    assert verdict["grounding_value_score"] == 0
    assert verdict["acceptance_gate_value"] == 0
    assert verdict["verdict_class"] == "null"
    assert verdict["honest_verdict"].startswith("complete_null")


def test_scenario_verify_7197_audit_request_denies_labels(
    sources: dict[str, object], predictions: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-7197-AUDIT rejects label transport into the child."""

    request = exp.build_audit_request(
        REPO, predictions, exp.build_shuffle_map(sources["authority"])
    )
    assert auditor.validate_request(request) == []
    leaked = deepcopy(request)
    leaked["support_label"] = "supported"
    assert "forbidden_label_transport" in auditor.validate_request(leaked)
    missing = deepcopy(request)
    missing.pop("candidate_predictions")
    assert "request_fields" in auditor.validate_request(missing)


def test_scenario_verify_7197_audit_recomputes_controls(
    built: tuple[dict[str, object], dict[str, Path]],
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7197-AUDIT runs fresh and keeps three controls per row."""

    artifact, paths = built
    rows = artifact["cold_audit_rows"]
    assert len(rows) == 128
    assert {control["intervention"] for row in rows for control in row["controls"]} == {
        "argument_reversal",
        "semantic_deletion",
        "shuffled_source",
    }
    assert artifact["independent_audit"]["candidate_module_imported"] is False
    assert artifact["independent_audit"]["decision_disagreement_count"] == 0
    assert artifact["independent_audit"]["labels_opened_after_prediction"] is True
    assert artifact["independent_audit"]["label_mutation_prediction_invariant"] is True
    assert paths["audit_request"].is_file()
    assert paths["audit_result"].is_file()
    request = json.loads(paths["audit_request"].read_text(encoding="utf-8"))
    direct = auditor.run_audit(request)
    assert direct["independent_semantic_audit_passed"] is True
    copied = tmp_path / "direct-audit.json"
    auditor._atomic_write_json(copied, direct)
    assert json.loads(copied.read_text(encoding="utf-8")) == direct


def test_scenario_verify_7197_preflight_checks_null_and_quarantine(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7197-PREFLIGHT checks readiness without promoting nulls."""

    artifact, _ = built
    checks = {row["check"]: row for row in artifact["preconditions_checked"]}
    assert checks["exp7195_structured_quarantine"]["observed_value"] is False
    assert checks["exp7196_structured_quarantine"]["observed_value"] is False
    assert checks["upstream_manifest_quarantine"]["observed_value"] == []
    assert checks["known_failed_value_not_promoted"]["observed_value"] == {
        "exp7182_grounding_value_score": 0,
        "exp7196_verdict_class": "null",
        "promoted_as_value": False,
    }
    assert checks["known_failed_value_not_promoted"]["passed"] is True


def test_scenario_verify_7197_preflight_writes_terminal_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7197-PREFLIGHT emits the full schema on a bad date."""

    checkpoint = tmp_path / "checkpoints" / "running.json"
    checkpoint.parent.mkdir()
    artifact = exp.build_artifact(
        REPO,
        "19000101",
        result_path=tmp_path / "blocked.json",
        checkpoint_path=checkpoint,
        audit_request_path=tmp_path / "checkpoints" / "request.json",
        audit_result_path=tmp_path / "checkpoints" / "response.json",
        duration_s=0.01,
    )
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "failed_check": "run_date",
        "upstream": "experiment_7197",
        "field": "run_date",
        "expected_value": exp.RUN_DATE,
        "observed_value": "19000101",
        "passed": False,
    }
    assert exp.validate_artifact(artifact, check_source_hashes=False) == []


def test_scenario_verify_7197_circularity_is_explicit(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7197-CIRCULARITY rejects a learned-verifier claim."""

    artifact, _ = built
    assert artifact["verifier_is_oracle"] is True
    assert (
        "same complete fixture correctness authority" in artifact["oracle_distinctness_rationale"]
    )
    assert "learned-verifier moat" in artifact["oracle_distinctness_rationale"]
    assert "32-base pilot" in artifact["scope_answer"]


def test_scenario_verify_7197_artifact_replays_and_rejects_tampering(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7197-ARTIFACT replays rows, gates, and checksums."""

    artifact, paths = built
    assert exp.validate_artifact(artifact, root=REPO, check_source_hashes=True) == []
    assert exp.validate_artifact(paths["result"], root=REPO, check_source_hashes=True) == []
    assert artifact["status"] == "complete"
    assert artifact["grounding_audit_complete_score"] == 1
    assert artifact["grounding_value_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"

    changed = deepcopy(artifact)
    changed["rows"].pop()
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "row_count" in exp.validate_artifact(changed, check_source_hashes=False)

    changed = deepcopy(artifact)
    changed["grounding_value_score"] = 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "terminal_classification" in exp.validate_artifact(changed, check_source_hashes=False)


def test_scenario_verify_7197_artifact_rejects_unreadable_and_wrong_schema() -> None:
    """SCENARIO-VERIFY-7197-ARTIFACT fails closed on malformed input."""

    assert exp.validate_artifact(Path("/does/not/exist")) == ["artifact_unreadable"]
    value = exp.base_artifact(REPO, exp.RUN_DATE)
    value.pop("rows")
    value["reproducibility_checksum"] = exp.artifact_checksum(value)
    errors = exp.validate_artifact(value, check_source_hashes=False)
    assert "artifact_fields" in errors


def test_scenario_verify_7197_artifact_hash_helpers_are_exact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7197-ARTIFACT binds exact bytes and stable JSON."""

    path = tmp_path / "bytes.bin"
    path.write_bytes(b"a\r\nb")
    assert exp.sha256_file(path) == exp.sha256_bytes(b"a\r\nb")
    assert exp.sha256_text("a\r\nb") == exp.sha256_bytes(b"a\r\nb")
    assert exp.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_scenario_verify_7197_policies_reject_bad_rosters_and_unknown_shapes(
    sources: dict[str, object], features: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-7197-POLICIES fails closed on malformed inputs."""

    public = sources["public"]
    calls = sources["capture"]["completion_rows"]
    with pytest.raises(ValueError, match="duplicate_call"):
        exp.build_candidate_features(public, [*calls, calls[0]])
    bad_public = deepcopy(public)
    bad_public[0]["label"] = "forbidden"
    with pytest.raises(ValueError, match="public_shape"):
        exp.build_candidate_features(bad_public, calls)
    with pytest.raises(ValueError, match="missing_call"):
        exp.build_candidate_features(public, calls[1:])
    extra = deepcopy(calls[0])
    extra["unit_id"] = "extra-unit"
    extra["call_id"] = "extra:source"
    with pytest.raises(ValueError, match="completion_call_roster"):
        exp.build_candidate_features(public, [*calls, extra])

    small_public, small_calls = _semantic_fixture()
    small_calls[0]["cache_hit"] = True
    small_calls[0]["reuse_from_call_id"] = "missing:source"
    with pytest.raises(ValueError, match="cache_origin_missing"):
        exp.build_candidate_features(small_public, small_calls)

    duplicate_authority = [deepcopy(sources["authority"][0])] * 2
    with pytest.raises(ValueError, match="shuffle_map_roster"):
        exp.build_shuffle_map(duplicate_authority)
    with pytest.raises(ValueError, match="shuffle_map_roster"):
        exp.evaluate_arms(features, {})

    valid = {
        "parse_status": "valid",
        "parsed": {"missing_fields": [], "relations": [], "entity_bindings": []},
    }
    assert (
        exp._typed_prediction("", {"parse_status": "valid", "parsed": None}, valid)["error"]
        == "parsed_output_missing"
    )
    missing = {"parse_status": "valid", "parsed": {"missing_fields": ["relation"]}}
    assert exp._typed_prediction("", missing, missing)["error"] == "declared_missing_fields"
    empty = {
        "parse_status": "valid",
        "parsed": {"missing_fields": [], "relations": [], "entity_bindings": []},
    }
    assert exp._typed_prediction("", empty, empty)["error"] == "claim_relation_count"
    _, semantic_calls = _semantic_fixture()
    semantic_calls[1]["raw_output"] = semantic_calls[1]["raw_output"].replace(
        '"surface":"entity-a"', '"surface":"missing-a"'
    )
    semantic_calls[1]["raw_output_sha256"] = exp.sha256_text(semantic_calls[1]["raw_output"])
    semantic_features = exp.build_candidate_features(small_public, semantic_calls)
    mapped = exp._typed_prediction(
        semantic_features[0]["source_text"],
        semantic_features[0]["source"],
        semantic_features[0]["claim"],
    )
    assert mapped["error"] == "entity_mapping:subject_id"
    assert exp._lexical_score("", "") == 0.0


def test_scenario_verify_7197_metrics_reject_invalid_scoring_and_bootstrap(
    sources: dict[str, object],
    predictions: list[dict[str, object]],
    scored: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7197-METRICS rejects incomplete scoring contracts."""

    with pytest.raises(ValueError, match="evaluation_authority_roster"):
        exp.score_evaluation(predictions, sources["authority"][:1])
    with pytest.raises(ValueError, match="prediction_roster"):
        exp.score_evaluation(predictions[:-1], sources["authority"])
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "evaluation_row_errors", lambda _: ["forced_invariant"])
        with pytest.raises(ValueError, match="evaluation_rows_invalid:forced_invariant"):
            exp.score_evaluation(predictions, sources["authority"])
    with pytest.raises(ValueError, match="cannot_summarize_invalid_rows"):
        exp.summarize_arms(scored[:-1])
    with pytest.raises(ValueError, match="cannot_bootstrap_invalid_rows"):
        exp.paired_cluster_bootstrap(scored[:-1], exp.BOOTSTRAP_SEED)
    with pytest.raises(ValueError, match="bootstrap_draws_positive"):
        exp.paired_cluster_bootstrap(scored, exp.BOOTSTRAP_SEED, draws=0)
    with pytest.raises(ValueError, match="paired_interval_missing"):
        exp._interval_lookup([], "typed_vs_direct", "accuracy")

    uneven = deepcopy(scored)
    moved_base = next(row["base_id"] for row in uneven if row["relation_family"] == "starts before")
    for row in uneven:
        if row["base_id"] == moved_base:
            row["relation_family"] = "ends before"
    assert exp.evaluation_row_errors(uneven) == []
    with pytest.raises(ValueError, match="bootstrap_family_cluster_count"):
        exp.paired_cluster_bootstrap(uneven, exp.BOOTSTRAP_SEED, draws=1)

    duplicated = deepcopy(scored)
    duplicated[1]["unit_id"] = duplicated[0]["unit_id"]
    errors = exp.evaluation_row_errors(duplicated)
    assert "baseline_unit_roster" in errors
    changed_order = deepcopy(scored)
    changed_order[0]["row_order"] = 1
    assert "direct_judgment:row_order" in exp.evaluation_row_errors(changed_order)
    changed_variant = deepcopy(scored)
    changed_variant[0]["variant"] = "wrong"
    assert "base_variant_roster" in exp.evaluation_row_errors(changed_variant)
    changed_family = deepcopy(scored)
    for row in changed_family:
        if row["arm"] == "direct_judgment" and row["relation_family"] == "starts before":
            row["relation_family"] = "ends before"
    assert "evaluation_families" in exp.evaluation_row_errors(changed_family)


def test_scenario_verify_7197_value_rejects_failed_independent_audit(
    scored: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7197-CIRCULARITY disqualifies evaluator disagreement."""

    metrics = exp.summarize_arms(scored)
    intervals = exp.paired_cluster_bootstrap(scored, exp.BOOTSTRAP_SEED, draws=5)
    verdict = exp.classify_value(metrics, intervals, independent_audit_passed=False)
    assert verdict["verdict_class"] == "disqualified"
    positive = deepcopy(verdict)
    positive["grounding_value_score"] = 1
    assert exp._scope_answer(metrics, positive).startswith("The typed policy passed")


def test_scenario_verify_7197_auditor_defensive_paths(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7197-AUDIT rejects malformed child requests and calls."""

    _, paths = built
    request = json.loads(paths["audit_request"].read_text(encoding="utf-8"))
    invalid = deepcopy(request)
    invalid["source_paths"] = {}
    invalid["candidate_predictions"] = "wrong"
    assert {"source_paths", "candidate_predictions"} <= set(auditor.validate_request(invalid))
    nested = deepcopy(request)
    nested["candidate_predictions"][0]["nested"] = [{"truth": "supported"}]
    assert "forbidden_label_transport" in auditor.validate_request(nested)
    with pytest.raises(ValueError, match="audit_request_invalid"):
        auditor.run_audit(invalid)

    wrong_hash = deepcopy(request)
    wrong_hash["expected_input_hashes"]["public_view"] = "sha256:wrong"
    with pytest.raises(ValueError, match="audit_input_hash_mismatch"):
        auditor.run_audit(wrong_hash)
    wrong_candidate = deepcopy(request)
    wrong_candidate["candidate_prediction_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="candidate_prediction_checksum_mismatch"):
        auditor.run_audit(wrong_candidate)

    public, calls = _semantic_fixture(reverse_claim=True)
    with pytest.raises(ValueError, match="duplicate_call"):
        auditor._build_features(public, [*calls, calls[0]])
    with pytest.raises(ValueError, match="missing_call"):
        auditor._build_features(public, calls[1:])
    bad_call = deepcopy(calls[0])
    bad_call["raw_output_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="raw_output_hash_mismatch"):
        auditor._call_features(bad_call)
    audit_features = auditor._build_features(public, calls)
    semantic_rows = auditor._prediction_rows(audit_features, {"unit-small": "unit-small"})
    assert semantic_rows[0]["prediction"] == "unsupported"
    assert {row["prediction"] for row in semantic_rows[0]["controls"]} >= {
        "supported",
        "abstain",
    }

    valid = audit_features[0]["source"]
    assert (
        auditor._typed_prediction("", {"parse_status": "valid", "parsed": None}, valid)["error"]
        == "parsed_output_missing"
    )
    missing = {"parse_status": "valid", "parsed": {"missing_fields": ["x"]}}
    assert auditor._typed_prediction("", missing, missing)["error"] == "declared_missing_fields"
    empty = {
        "parse_status": "valid",
        "parsed": {"missing_fields": [], "relations": [], "entity_bindings": []},
    }
    assert auditor._typed_prediction("", empty, empty)["error"] == "claim_relation_count"
    broken = deepcopy(audit_features[0]["claim"])
    broken["parsed"]["entity_bindings"][0]["surface"] = "missing"
    assert auditor._typed_prediction(
        audit_features[0]["source_text"], audit_features[0]["source"], broken
    )["error"].startswith("entity_mapping:")

    disagreement = deepcopy(request)
    disagreement["candidate_predictions"][64]["prediction"] = "supported"
    disagreement["candidate_prediction_checksum"] = auditor._candidate_checksum(
        disagreement["candidate_predictions"]
    )
    result = auditor.run_audit(disagreement)
    assert result["independent_semantic_audit_passed"] is False
    assert result["decision_disagreement_count"] == 1


def test_scenario_verify_7197_preflight_rejects_each_external_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7197-PREFLIGHT covers each fail-closed source gate."""

    outputs = [tmp_path / name for name in ("result", "checkpoint", "request", "response")]

    def run(
        overrides: dict[str, Path] | None = None,
        pins: dict[str, str] | None = None,
        selected_outputs: list[Path] | None = None,
    ) -> list[dict[str, object]]:
        paths = exp._resolve_paths(REPO, overrides)
        checks, _ = exp._preconditions(
            REPO,
            exp.RUN_DATE,
            paths,
            selected_outputs or outputs,
            pinned_hashes=pins,
        )
        return checks

    assert run({"module": tmp_path / "missing"})[-1]["check"] == "module_path"
    changed_public = tmp_path / "public.jsonl"
    changed_public.write_bytes(PUBLIC.read_bytes() + b"\n")
    assert run({"public_view": changed_public})[-1]["check"] == "public_view_hash"

    missing_spec = tmp_path / "spec.md"
    missing_spec.write_text("no requirement", encoding="utf-8")
    assert run({"constraint_spec": missing_spec})[-1]["check"] == "constraint_spec_requirement"

    changed_7195 = json.loads(
        (REPO / exp.SOURCE_PATHS["exp7195_artifact"]).read_text(encoding="utf-8")
    )
    changed_7195["flagged_adversarial"] = True
    flagged_path = tmp_path / "flagged7195.json"
    flagged_path.write_text(json.dumps(changed_7195), encoding="utf-8")
    pins = dict(exp.PINNED_INPUT_HASHES)
    pins["exp7195_artifact"] = exp.sha256_file(flagged_path)
    assert run({"exp7195_artifact": flagged_path}, pins)[-1]["check"] == (
        "exp7195_structured_quarantine"
    )

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("- experiment_id: '7195'\n", encoding="utf-8")
    assert run({"exclusion_manifest": manifest})[-1]["check"] == ("upstream_manifest_quarantine")

    changed_7195.pop("flagged_adversarial")
    changed_7195["status"] = "partial"
    terminal_path = tmp_path / "terminal7195.json"
    terminal_path.write_text(json.dumps(changed_7195), encoding="utf-8")
    pins["exp7195_artifact"] = exp.sha256_file(terminal_path)
    assert run({"exp7195_artifact": terminal_path}, pins)[-1]["check"] == (
        "exp7195_terminal_gate_fields"
    )

    changed_7195["status"] = "complete"
    for row in changed_7195["error_decomposition_rows"]:
        row["old_grounding_value_score"] = 1
    value_path = tmp_path / "value7195.json"
    value_path.write_text(json.dumps(changed_7195), encoding="utf-8")
    pins["exp7195_artifact"] = exp.sha256_file(value_path)
    assert run({"exp7195_artifact": value_path}, pins)[-1]["check"] == (
        "known_failed_value_not_promoted"
    )

    raw = json.loads((REPO / exp.SOURCE_PATHS["raw_manifest"]).read_text(encoding="utf-8"))
    raw["status"] = "partial"
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    raw_pins = dict(exp.PINNED_INPUT_HASHES)
    raw_pins["raw_manifest"] = exp.sha256_file(raw_path)
    assert run({"raw_manifest": raw_path}, raw_pins)[-1]["check"] == "raw_call_manifest"

    monkeypatch.setattr(exp.sys, "executable", "/bin/true")
    assert run()[-1]["check"] == "python_executable"
    monkeypatch.undo()
    missing_output = tmp_path / "missing-parent" / "result.json"
    assert run(selected_outputs=[missing_output])[-1]["check"].startswith("output_directory")
    assert exp._unwrap({"principle": "p", "value": True}) is True


def test_scenario_verify_7197_artifact_validation_reports_each_contract_defect(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7197-ARTIFACT detects every terminal contract defect."""

    artifact, _ = built

    def errors(field: str, value: object, *, checksum: bool = True) -> list[str]:
        changed = deepcopy(artifact)
        changed[field] = value
        if checksum:
            changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed, check_source_hashes=False)

    cases = (
        ("field_principles", {}, "field_principles"),
        ("run_date", "wrong", "run_date"),
        ("random_seed", 0, "random_seed"),
        ("MODEL_SPECS", [{}], "model_invocation"),
        ("status", "running", "terminal_status"),
        ("inference_substrate", "wrong", "inference_substrate"),
        ("inference_substrate_class", "aggregation", "inference_substrate_class"),
        ("arm_metrics", {}, "arm_metrics"),
        ("paired_interval_rows", [], "paired_interval_rows"),
        ("scope_answer", "wrong", "scope_answer"),
        ("cold_audit_rows", [], "cold_audit_rows"),
        ("oracle_distinctness_rationale", "wrong", "oracle_distinctness_rationale"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("grounding_audit_complete_score", 0, "grounding_audit_complete_score"),
        ("source_artifact_hashes", {}, "source_artifact_hashes"),
    )
    for field, value, expected in cases:
        check_sources = field == "source_artifact_hashes"
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        observed = exp.validate_artifact(changed, root=REPO, check_source_hashes=check_sources)
        assert expected in observed
    assert "reproducibility_checksum" in errors(
        "reproducibility_checksum", "sha256:wrong", checksum=False
    )

    for field, expected in (
        ("candidate_module_imported", "candidate_module_imported"),
        ("labels_opened_after_prediction", "label_open_order"),
        ("label_mutation_prediction_invariant", "label_mutation_prediction_invariant"),
        ("decision_disagreement_count", "audit_decision_disagreement"),
        ("controls_complete", "audit_controls"),
    ):
        changed = deepcopy(artifact)
        changed["independent_audit"][field] = (
            1 if field == "decision_disagreement_count" else field == "candidate_module_imported"
        )
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, check_source_hashes=False)

    changed = deepcopy(artifact)
    changed["preconditions_checked"][0]["passed"] = False
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "preconditions_checked" in exp.validate_artifact(changed, check_source_hashes=False)

    blocked = deepcopy(artifact)
    blocked.update(
        {
            "status": "blocked",
            "verdict_class": "positive",
            "grounding_audit_complete_score": 1,
            "inference_substrate_class": "aggregation",
            "gate_check_summary": {"passed": True},
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked, check_source_hashes=False)
    assert {
        "blocked_verdict",
        "blocked_completion",
        "blocked_substrate_class",
        "blocked_gate_summary",
    } <= set(blocked_errors)


def test_scenario_verify_7197_artifact_refuses_failed_final_cold_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7197-ARTIFACT never writes a failed terminal candidate."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced_invalid"])
    with pytest.raises(ValueError, match="terminal_artifact_invalid:forced_invalid"):
        exp.build_artifact(
            REPO,
            exp.RUN_DATE,
            result_path=tmp_path / "result.json",
            checkpoint_path=checkpoint_dir / "running.json",
            audit_request_path=checkpoint_dir / "request.json",
            audit_result_path=checkpoint_dir / "response.json",
            duration_s=0.01,
        )
    assert not (tmp_path / "result.json").exists()
